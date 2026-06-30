import { lstat, mkdir } from "node:fs/promises";
import { dirname, isAbsolute, parse, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const SRC_DIR = dirname(fileURLToPath(import.meta.url));

export const PACKAGE_ROOT = resolve(SRC_DIR, "..");
export const REPO_ROOT = resolve(PACKAGE_ROOT, "../..");
export const SHADOW_OUTPUT_ROOTS = [
  resolve(PACKAGE_ROOT, "artifacts", "control_shadow"),
  resolve("/tmp", "wca-exp-control-ts"),
] as const;

export function resolveFromPackageRoot(path: string): string {
  return isAbsolute(path) ? resolve(path) : resolve(PACKAGE_ROOT, path);
}

export function shadowOutputRestrictionMessage(label: string): string {
  return `${label} is restricted to shadow-only directories: ${SHADOW_OUTPUT_ROOTS.join(", ")}`;
}

export async function prepareShadowOutputDirectory(path: string, label: string): Promise<string> {
  const resolved = assertAllowedShadowOutputPath(path, label);
  await rejectSymlinkComponents(resolved, label);
  await mkdir(resolved, { recursive: true });
  await rejectSymlinkComponents(resolved, label);
  return resolved;
}

export async function prepareShadowOutputFile(path: string, label: string): Promise<string> {
  const resolved = assertAllowedShadowOutputPath(path, label);
  const parent = dirname(resolved);
  await rejectSymlinkComponents(parent, label);
  await mkdir(parent, { recursive: true });
  await rejectSymlinkComponents(parent, label);
  await rejectExistingSymlink(resolved, label);
  return resolved;
}

function assertAllowedShadowOutputPath(path: string, label: string): string {
  const resolved = resolveFromPackageRoot(path);
  if (!SHADOW_OUTPUT_ROOTS.some((root) => isPathInside(resolved, root))) {
    throw new Error(shadowOutputRestrictionMessage(label));
  }
  return resolved;
}

function isPathInside(candidate: string, root: string): boolean {
  const relativePath = relative(root, candidate);
  return relativePath === "" || (relativePath.length > 0 && !relativePath.startsWith("..") && !relativePath.startsWith("/"));
}

async function rejectSymlinkComponents(path: string, label: string): Promise<void> {
  const resolved = resolve(path);
  const root = parse(resolved).root;
  const parts = resolved.slice(root.length).split(sep).filter(Boolean);
  let current = root;
  for (const part of parts) {
    current = resolve(current, part);
    if (isAllowedPlatformTmpLink(current, resolved)) {
      continue;
    }
    let stat;
    try {
      stat = await lstat(current);
    } catch (error) {
      if (error instanceof Error && "code" in error && error.code === "ENOENT") {
        break;
      }
      throw error;
    }
    if (stat.isSymbolicLink()) {
      throw new Error(`${label} contains symlinked path component: ${current}`);
    }
  }
}

async function rejectExistingSymlink(path: string, label: string): Promise<void> {
  try {
    const stat = await lstat(path);
    if (stat.isSymbolicLink()) {
      throw new Error(`${label} contains symlinked path component: ${path}`);
    }
  } catch (error) {
    if (error instanceof Error && "code" in error && error.code === "ENOENT") {
      return;
    }
    throw error;
  }
}

function isAllowedPlatformTmpLink(component: string, fullPath: string): boolean {
  return component === resolve("/tmp") && isPathInside(fullPath, resolve("/tmp", "wca-exp-control-ts"));
}
