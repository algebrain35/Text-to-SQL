//Reads saved settings from localStorage and returns null if storage is unavailable
export function readSetting(key: string): string | null {
  try {
    return window.localStorage.getItem(`schemagraph.${key}`);
  } catch {
    return null;
  }
}

//Saves settings to localStorage
export function writeSetting(key: string, value: string): void {
  try {
    window.localStorage.setItem(`schemagraph.${key}`, value);
  } catch {
    //Ignore storage errors since the application can still function without saving
  }
}
