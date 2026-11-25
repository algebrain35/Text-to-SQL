/*
Lets the browse database tables, view their columns and constriants
and display declared and inferred relationships between tables 
*/

import { useEffect, useState } from "react";
import { getTable, listTables, type RelationshipMeta, type TableSummary, type TableWithRelationships } from "../api/client";
import { displayName } from "../lib/names";
import { errorMessage } from "./Notices";

function Relationship({ rel }: { rel: RelationshipMeta }) {
  const ev = rel.evidence;
  return (
    <li className={rel.source}>
      <span className="ident small">
        {displayName(rel.from_column)} = {displayName(rel.to_column)}
      </span>{" "}
      {rel.source === "declared" ? (
        <span className="hint">declared, {rel.cardinality}</span>
      ) : (
        <span className="inferred-note">
          inferred, confidence {rel.confidence.toFixed(2)}
          {ev ? ` (${ev.name_match ? `${ev.name_match.replace("_", " ")} name match, ` : ""}${Math.round(ev.value_overlap * 100)}% of values found)` : ""}
        </span>
      )}
    </li>
  );
}

function TableDetail({ detail }: { detail: TableWithRelationships }) {
  //Creates a row for each column in the table
  const columnRows = detail.columns.map((c) => {
    const notes: string[] = [];

    //Adds notes about the column's constraints
    if (c.is_primary_key) {
      notes.push("primary key");
    }

    if (c.is_unique && !c.is_primary_key) {
      notes.push("unique");
    }

    if (c.is_nullable) {
      notes.push("nullable");
    }

    const columnNotes = notes.join(", ");

    //Returns a row containing the column information
    return (
      <tr key={c.uid}>
        <td className="ident">{c.name}</td>
        <td>{c.data_type}</td>
        <td>{columnNotes}</td>
      </tr>
    );
  });

  //Gets the outgoing and incoming relationsships
  const references = detail.references ?? [];
  const referencedBy = detail.referenced_by ?? [];

  //Initializes empty relationship displays
  let referencesDisplay = <></>;
  let referencedByDisplay = <></>;

  //Displays relationships pointing to other tables
  if (references.length > 0) {
    const relationshipList = references.map((r, i) => (
      <Relationship key={i} rel={r} />
    ));

    referencesDisplay = (
      <>
        <h4>Points to</h4>
        <ul className="plain">{relationshipList}</ul>
      </>
    );
  }

  //Displays relationships pointing to the selected table
  if (referencedBy.length > 0) {
    const relationshipList = referencedBy.map((r, i) => (
      <Relationship key={i} rel={r} />
    ));

    referencedByDisplay = (
      <>
        <h4>Pointed to by</h4>
        <ul className="plain">{relationshipList}</ul>
      </>
    );
  }

  return (
    <div className="table-detail">
      <h4 className="ident">{displayName(detail.table.uid)}</h4>
      {detail.table.description && <p>{detail.table.description}</p>}
      <div className="table-scroll" tabIndex={0} role="region" aria-label="Columns">
        <table>
          <thead>
            <tr>
              <th scope="col">Column</th>
              <th scope="col">Type</th>
              <th scope="col">Notes</th>
            </tr>
          </thead>
          <tbody>
            {columnRows}
          </tbody>
        </table>
      </div>
      {referencesDisplay}
      {referencedByDisplay}
    </div>
  );
}

//Allows the user to browse database tables and their relationships when opened
export function SchemaBrowser({ dbId }: { dbId: string }) {
  const [open, setOpen] = useState(false);
  const [tables, setTables] = useState<TableSummary[] | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<TableWithRelationships | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

  useEffect(() => {
    setTables(null);
    setSelected(null);
    setDetail(null);
    setProblem(null);
  }, [dbId]);

  useEffect(() => {
    if (!open || !dbId || tables) return;
    const controller = new AbortController();
    listTables(dbId, controller.signal)
      .then(setTables)
      .catch((e) => {
        if (!controller.signal.aborted) setProblem(`${errorMessage(e).title}. ${errorMessage(e).body}`);
      });
    return () => controller.abort();
  }, [open, dbId, tables]);

  useEffect(() => {
    if (!selected) return;
    const controller = new AbortController();
    setDetail(null);
    getTable(dbId, selected, controller.signal)
      .then(setDetail)
      .catch((e) => {
        if (!controller.signal.aborted) setProblem(`${errorMessage(e).title}. ${errorMessage(e).body}`);
      });
    return () => controller.abort();
  }, [dbId, selected]);

  return (
    <details className="browser" onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>Browse the tables in {dbId || "this database"}</summary>
      {problem && <p className="error-text">{problem}</p>}
      {open && !tables && !problem && <p className="hint">Loading tables</p>}
      {tables && (
        <div className="browser-body">
          <ul className="table-list">
            {tables.map((t) => (
              <li key={t.uid}>
                <button type="button" className="quiet" aria-pressed={selected === t.uid} onClick={() => setSelected(t.uid)}>
                  <span className="ident">{t.display_name}</span>
                  <span className="hint"> {t.column_count} columns</span>
                </button>
                {t.description && <p className="hint">{t.description}</p>}
              </li>
            ))}
          </ul>
          {detail && <TableDetail detail={detail} />}
        </div>
      )}
    </details>
  );
}
