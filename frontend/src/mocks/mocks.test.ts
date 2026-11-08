import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import askMock from "./ask_response.json";
import joinPlanMock from "./join_plan.json";
import metadataMock from "./metadata.adventureworks.json";

// The mocks are copies of contracts/examples (SPEC 5.22). If the contract changes, copy the files again.
const example = (name: string) => JSON.parse(readFileSync(resolve(__dirname, "../../../contracts/examples", name), "utf8"));

describe("mocks", () => {
  it.each([
    ["ask_response.sample.json", askMock],
    ["join_plan.sample.json", joinPlanMock],
    ["metadata.adventureworks.sample.json", metadataMock],
  ])("%s matches contracts/examples", (name, copy) => {
    expect(copy).toEqual(example(name));
  });
});
