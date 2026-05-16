package middleware

import (
	"context"
	"errors"
	"log/slog"
	"net"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgtype"
	"github.com/multica-ai/multica/server/internal/auth"
	"github.com/multica-ai/multica/server/internal/util"
	db "github.com/multica-ai/multica/server/pkg/db/generated"
)

func uuidToString(u pgtype.UUID) string { return util.UUIDToString(u) }

// Auth middleware validates JWT tokens or Personal Access Tokens.
// Token sources (in priority order):
//  1. Authorization: Bearer <token> header (PAT or JWT)
//  2. multica_auth HttpOnly cookie (JWT) — requires valid CSRF token for state-changing requests
//
// Sets X-User-ID and X-User-Email headers on the request for downstream handlers.
//
// patCache is optional; when non-nil, PAT lookups are cached with a short
// TTL (auth.AuthCacheTTL). On cache hit the middleware skips both the DB
// SELECT and the last_used_at UPDATE — last_used_at is therefore refreshed
// at most once per TTL window per token, not per request.
func Auth(queries *db.Queries, patCache *auth.PATCache) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if trustedProxyAuthEnabled() && trustedProxyRequest(r) {
				user, workspaceSlug, ok := authenticateTrustedProxy(w, r, queries)
				if !ok {
					return
				}
				r.Header.Set("X-User-ID", uuidToString(user.ID))
				r.Header.Set("X-User-Email", user.Email)
				setTrustedProxySessionCookies(w, workspaceSlug)
				next.ServeHTTP(w, r)
				return
			}

			tokenString, fromCookie := extractToken(r)
			if tokenString == "" {
				slog.Debug("auth: no token found", "path", r.URL.Path)
				http.Error(w, `{"error":"missing authorization"}`, http.StatusUnauthorized)
				return
			}

			// Cookie-based auth requires CSRF validation for state-changing methods.
			if fromCookie && !auth.ValidateCSRF(r) {
				slog.Debug("auth: CSRF validation failed", "path", r.URL.Path)
				http.Error(w, `{"error":"CSRF validation failed"}`, http.StatusForbidden)
				return
			}

			// PAT: tokens starting with "mul_"
			if strings.HasPrefix(tokenString, "mul_") {
				hash := auth.HashToken(tokenString)

				// Cache hit: TTL has not expired, the token was valid the
				// last time we looked, and nothing has invalidated the
				// entry since. Skip the DB SELECT and the last_used_at
				// UPDATE — last_used_at is bumped once per TTL window.
				if userID, ok := patCache.Get(r.Context(), hash); ok {
					r.Header.Set("X-User-ID", userID)
					next.ServeHTTP(w, r)
					return
				}

				if queries == nil {
					http.Error(w, `{"error":"invalid token"}`, http.StatusUnauthorized)
					return
				}
				pat, err := queries.GetPersonalAccessTokenByHash(r.Context(), hash)
				if err != nil {
					slog.Warn("auth: invalid PAT", "path", r.URL.Path, "error", err)
					http.Error(w, `{"error":"invalid token"}`, http.StatusUnauthorized)
					return
				}

				userID := uuidToString(pat.UserID)
				r.Header.Set("X-User-ID", userID)

				// Clamp cache TTL to the token's remaining lifetime so a
				// PAT expiring in <AuthCacheTTL can't continue passing
				// auth on a cache hit after expires_at.
				var expiresAt time.Time
				if pat.ExpiresAt.Valid {
					expiresAt = pat.ExpiresAt.Time
				}
				patCache.Set(r.Context(), hash, userID, auth.TTLForExpiry(time.Now(), expiresAt))

				// Cache miss = TTL expired (or first use after revoke /
				// process restart). Refresh last_used_at; subsequent hits
				// within the TTL window skip this write entirely.
				go queries.UpdatePersonalAccessTokenLastUsed(context.Background(), pat.ID)

				next.ServeHTTP(w, r)
				return
			}

			// JWT
			token, err := jwt.Parse(tokenString, func(token *jwt.Token) (any, error) {
				if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
					return nil, jwt.ErrSignatureInvalid
				}
				return auth.JWTSecret(), nil
			})
			if err != nil || !token.Valid {
				slog.Warn("auth: invalid token", "path", r.URL.Path, "error", err)
				http.Error(w, `{"error":"invalid token"}`, http.StatusUnauthorized)
				return
			}

			claims, ok := token.Claims.(jwt.MapClaims)
			if !ok {
				slog.Warn("auth: invalid claims", "path", r.URL.Path)
				http.Error(w, `{"error":"invalid claims"}`, http.StatusUnauthorized)
				return
			}

			sub, ok := claims["sub"].(string)
			if !ok || strings.TrimSpace(sub) == "" {
				slog.Warn("auth: invalid claims", "path", r.URL.Path)
				http.Error(w, `{"error":"invalid claims"}`, http.StatusUnauthorized)
				return
			}
			r.Header.Set("X-User-ID", sub)
			if email, ok := claims["email"].(string); ok {
				r.Header.Set("X-User-Email", email)
			}

			next.ServeHTTP(w, r)
		})
	}
}

func trustedProxyAuthEnabled() bool {
	v := strings.ToLower(strings.TrimSpace(os.Getenv("MULTICA_TRUSTED_PROXY_AUTH")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func trustedProxyRequest(r *http.Request) bool {
	if r.Header.Get("X-Horde-Trusted") == "1" {
		return true
	}
	v := strings.ToLower(strings.TrimSpace(os.Getenv("MULTICA_TRUSTED_PROXY_ALWAYS")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func authenticateTrustedProxy(w http.ResponseWriter, r *http.Request, queries *db.Queries) (db.User, string, bool) {
	var zero db.User
	if queries == nil {
		http.Error(w, `{"error":"trusted proxy auth requires database access"}`, http.StatusInternalServerError)
		return zero, "", false
	}
	if !isLoopbackRemote(r.RemoteAddr) {
		slog.Warn("auth: rejected trusted proxy header from non-loopback peer", "remote", r.RemoteAddr, "path", r.URL.Path)
		http.Error(w, `{"error":"trusted proxy auth rejected"}`, http.StatusForbidden)
		return zero, "", false
	}

	email := firstNonEmpty(
		r.Header.Get("X-Horde-Email"),
		os.Getenv("MULTICA_TRUSTED_PROXY_EMAIL"),
		"horde@local.multica",
	)
	name := firstNonEmpty(
		r.Header.Get("X-Horde-Name"),
		os.Getenv("MULTICA_TRUSTED_PROXY_NAME"),
		strings.Split(email, "@")[0],
		"Horde User",
	)
	workspaceName := firstNonEmpty(
		r.Header.Get("X-Horde-Workspace"),
		os.Getenv("MULTICA_TRUSTED_PROXY_WORKSPACE"),
		"Horde",
	)
	workspaceSlug := slugify(firstNonEmpty(
		r.Header.Get("X-Horde-Workspace-Slug"),
		os.Getenv("MULTICA_TRUSTED_PROXY_WORKSPACE_SLUG"),
		workspaceName,
		"horde",
	))

	user, err := getOrCreateTrustedProxyUser(r.Context(), queries, name, email)
	if err != nil {
		slog.Error("auth: trusted proxy user setup failed", "path", r.URL.Path, "error", err)
		http.Error(w, `{"error":"trusted proxy user setup failed"}`, http.StatusInternalServerError)
		return zero, "", false
	}
	if err := ensureTrustedProxyWorkspace(r.Context(), queries, user.ID, workspaceName, workspaceSlug); err != nil {
		slog.Error("auth: trusted proxy workspace setup failed", "path", r.URL.Path, "error", err)
		http.Error(w, `{"error":"trusted proxy workspace setup failed"}`, http.StatusInternalServerError)
		return zero, "", false
	}
	return user, workspaceSlug, true
}

func getOrCreateTrustedProxyUser(ctx context.Context, queries *db.Queries, name, email string) (db.User, error) {
	user, err := queries.GetUserByEmail(ctx, email)
	if err == nil {
		return user, nil
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return db.User{}, err
	}
	user, err = queries.CreateUser(ctx, db.CreateUserParams{
		Name:      name,
		Email:     email,
		AvatarUrl: pgtype.Text{},
	})
	if err == nil {
		return user, nil
	}
	// Concurrent first requests can race on the unique email constraint.
	return queries.GetUserByEmail(ctx, email)
}

func ensureTrustedProxyWorkspace(ctx context.Context, queries *db.Queries, userID pgtype.UUID, name, slug string) error {
	workspace, err := queries.GetWorkspaceBySlug(ctx, slug)
	if errors.Is(err, pgx.ErrNoRows) {
		workspace, err = queries.CreateWorkspace(ctx, db.CreateWorkspaceParams{
			Name:        name,
			Slug:        slug,
			Description: pgtype.Text{},
			Context:     pgtype.Text{},
			IssuePrefix: issuePrefix(name),
		})
		if err != nil {
			workspace, err = queries.GetWorkspaceBySlug(ctx, slug)
		}
	}
	if err != nil {
		return err
	}

	_, err = queries.GetMemberByUserAndWorkspace(ctx, db.GetMemberByUserAndWorkspaceParams{
		UserID:      userID,
		WorkspaceID: workspace.ID,
	})
	if errors.Is(err, pgx.ErrNoRows) {
		_, err = queries.CreateMember(ctx, db.CreateMemberParams{
			WorkspaceID: workspace.ID,
			UserID:      userID,
			Role:        "owner",
		})
		if err != nil {
			_, err = queries.GetMemberByUserAndWorkspace(ctx, db.GetMemberByUserAndWorkspaceParams{
				UserID:      userID,
				WorkspaceID: workspace.ID,
			})
		}
	}
	if err != nil {
		return err
	}
	_, err = queries.MarkUserOnboarded(ctx, userID)
	return err
}

func isLoopbackRemote(remoteAddr string) bool {
	host, _, err := net.SplitHostPort(remoteAddr)
	if err != nil {
		host = remoteAddr
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

func setTrustedProxySessionCookies(w http.ResponseWriter, workspaceSlug string) {
	const oneYear = 60 * 60 * 24 * 365
	http.SetCookie(w, &http.Cookie{
		Name:     "multica_logged_in",
		Value:    "1",
		Path:     "/",
		MaxAge:   oneYear,
		SameSite: http.SameSiteLaxMode,
	})
	http.SetCookie(w, &http.Cookie{
		Name:     "last_workspace_slug",
		Value:    workspaceSlug,
		Path:     "/",
		MaxAge:   oneYear,
		SameSite: http.SameSiteLaxMode,
	})
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if s := strings.TrimSpace(v); s != "" {
			return s
		}
	}
	return ""
}

func slugify(value string) string {
	var b strings.Builder
	lastDash := false
	for _, r := range strings.ToLower(value) {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
			lastDash = false
			continue
		}
		if !lastDash && b.Len() > 0 {
			b.WriteByte('-')
			lastDash = true
		}
	}
	slug := strings.Trim(b.String(), "-")
	if slug == "" {
		return "horde"
	}
	if len(slug) > 48 {
		slug = strings.Trim(slug[:48], "-")
	}
	if slug == "" {
		return "horde"
	}
	return slug
}

func issuePrefix(name string) string {
	var b strings.Builder
	for _, r := range strings.ToUpper(name) {
		if (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
			if b.Len() >= 3 {
				return b.String()
			}
		}
	}
	for b.Len() < 3 {
		b.WriteByte('X')
	}
	return b.String()
}

// extractToken returns the bearer token and whether it came from a cookie.
// Priority: Authorization header > multica_auth cookie.
func extractToken(r *http.Request) (token string, fromCookie bool) {
	if authHeader := r.Header.Get("Authorization"); authHeader != "" {
		tokenString := strings.TrimPrefix(authHeader, "Bearer ")
		if tokenString != authHeader {
			return tokenString, false
		}
	}

	if cookie, err := r.Cookie(auth.AuthCookieName); err == nil && cookie.Value != "" {
		return cookie.Value, true
	}

	return "", false
}
