const rawBasePath = process.env.NEXT_PUBLIC_BASE_PATH ?? process.env.MULTICA_WEB_BASE_PATH ?? "";

export const publicBasePath = rawBasePath.replace(/\/$/, "");

export function publicPath(path: string): string {
  if (!path.startsWith("/")) return `${publicBasePath}/${path}`;
  return `${publicBasePath}${path}`;
}
