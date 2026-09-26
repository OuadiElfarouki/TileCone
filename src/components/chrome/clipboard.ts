/**
 * Copy text in secure and plain-HTTP deployments.
 *
 * The asynchronous Clipboard API is preferred, but browsers expose it only in
 * secure contexts. The temporary textarea keeps local-network beta builds
 * useful; callers receive a real result so a failed copy is never silent.
 */
export async function copyText(text: string): Promise<boolean> {
  try {
    if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // A denied Clipboard API may still permit the legacy selection command.
  }

  if (typeof document === "undefined" || !document.body) return false;
  const field = document.createElement("textarea");
  field.value = text;
  field.setAttribute("readonly", "");
  field.style.position = "fixed";
  field.style.opacity = "0";
  field.style.pointerEvents = "none";
  const previousFocus = document.activeElement as HTMLElement | null;
  document.body.appendChild(field);
  field.focus();
  field.select();
  try {
    return document.execCommand("copy");
  } catch {
    return false;
  } finally {
    field.remove();
    previousFocus?.focus();
  }
}
