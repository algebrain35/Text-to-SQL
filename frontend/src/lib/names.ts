//Removes the database ID from the UID for display
//Keeps table and column names intact because they may contain dots
export function displayName(uid: string): string {
  const i = uid.indexOf(":");
  return i < 0 ? uid : uid.slice(i + 1);
}
