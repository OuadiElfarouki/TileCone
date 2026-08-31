import { afterEach, describe, expect, it, vi } from "vitest";
import { copyText } from "../ui/clipboard";

afterEach(() => {
  vi.unstubAllGlobals();
});

function legacyDocument(result: boolean) {
  const field = {
    value: "",
    style: {} as Record<string, string>,
    setAttribute: vi.fn(),
    focus: vi.fn(),
    select: vi.fn(),
    remove: vi.fn(),
  };
  return {
    field,
    document: {
      body: { appendChild: vi.fn() },
      createElement: vi.fn(() => field),
      execCommand: vi.fn(() => result),
    },
  };
}

describe("copyText", () => {
  it("prefers the asynchronous Clipboard API", async () => {
    const writeText = vi.fn(async () => undefined);
    vi.stubGlobal("navigator", { clipboard: { writeText } });

    expect(await copyText("A[0:4]")).toBe(true);
    expect(writeText).toHaveBeenCalledWith("A[0:4]");
  });

  it("falls back to a selected textarea on plain HTTP", async () => {
    const legacy = legacyDocument(true);
    vi.stubGlobal("navigator", {});
    vi.stubGlobal("document", legacy.document);

    expect(await copyText("B[:, 2]")).toBe(true);
    expect(legacy.field.value).toBe("B[:, 2]");
    expect(legacy.field.select).toHaveBeenCalled();
    expect(legacy.document.execCommand).toHaveBeenCalledWith("copy");
    expect(legacy.field.remove).toHaveBeenCalled();
  });

  it("reports failure when neither copy path succeeds", async () => {
    const legacy = legacyDocument(false);
    vi.stubGlobal("navigator", { clipboard: { writeText: vi.fn(async () => { throw new Error("denied"); }) } });
    vi.stubGlobal("document", legacy.document);

    expect(await copyText("C[0]")).toBe(false);
  });
});
