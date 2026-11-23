/** localStorage that never throws (private windows, blocked storage). Used only for small conveniences. */
export function readSetting(key: string): string | null {
  try {
    return window.localStorage.getItem(`schemagraph.${key}`);
  } catch {
    return null;
  }
}

export function writeSetting(key: string, value: string): void {
  try {
    window.localStorage.setItem(`schemagraph.${key}`, value);
  } catch {
    /* not saved; the page works without it */
  }
}
