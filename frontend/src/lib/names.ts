/** Name shown to people for a UID (`db:schema.table[.column]`): the part after the db_id.
 *  A db_id never contains ":" (SPEC 2.1), so cutting at the first one is safe. Never cut a name further:
 *  table and column names can contain dots. */
export function displayName(uid: string): string {
  const i = uid.indexOf(":");
  return i < 0 ? uid : uid.slice(i + 1);
}
