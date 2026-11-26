import { RESEARCH_TYPES_NOT_RANDOM } from "@solaris/common";
import {
    NeutralBorderPosture,
    NotesDirectives,
    NotesParseResult,
    OperatingMode,
} from "./types";

const START_MARKER = /^#\s*===\s*AI\s*===\s*$/i;
const END_MARKER = /^#\s*===\s*END\s+AI\s*===\s*$/i;

const VALID_MODES = new Set<OperatingMode>([
    "paused",
    "advisor",
    "approval",
    "autonomous",
]);
const VALID_TECHS = new Set<string>(RESEARCH_TYPES_NOT_RANDOM);

export const defaultNotesDirectives = (): NotesDirectives => ({
    attack: [],
    expectTech: [],
});

const stripInlineComment = (raw: string): string => {
    let inSingle = false;
    let inDouble = false;
    for (let i = 0; i < raw.length; i++) {
        const ch = raw[i];
        if (ch === "'" && !inDouble) inSingle = !inSingle;
        else if (ch === '"' && !inSingle) inDouble = !inDouble;
        else if (ch === "#" && !inSingle && !inDouble) {
            return raw.slice(0, i).trim();
        }
    }
    return raw.trim();
};

const unquote = (s: string): string => {
    const t = s.trim();
    if (
        (t.startsWith('"') && t.endsWith('"')) ||
        (t.startsWith("'") && t.endsWith("'"))
    ) {
        return t.slice(1, -1).trim();
    }
    return t;
};

const parseStringList = (val: string): string[] | null => {
    const t = val.trim();
    if (!t || t === "[]" || t.toLowerCase() === "none") return [];
    let inner = t;
    if (t.startsWith("[") || t.endsWith("]")) {
        if (!t.startsWith("[") || !t.endsWith("]")) return null;
        inner = t.slice(1, -1).trim();
        if (!inner) return [];
    }
    const items: string[] = [];
    let cur = "";
    let inSingle = false;
    let inDouble = false;
    for (let i = 0; i < inner.length; i++) {
        const ch = inner[i];
        if (ch === "'" && !inDouble) {
            inSingle = !inSingle;
            cur += ch;
        } else if (ch === '"' && !inSingle) {
            inDouble = !inDouble;
            cur += ch;
        } else if (ch === "," && !inSingle && !inDouble) {
            const clean = unquote(cur);
            if (clean) items.push(clean);
            cur = "";
        } else {
            cur += ch;
        }
    }
    if (inSingle || inDouble) return null;
    const tail = unquote(cur);
    if (tail) items.push(tail);
    return items;
};

/**
 * Parses the optional `# === AI === ... # === END AI ===` block inside
 * Solaris's private In-Game Notes (`GET /api/game/:gameId/notes`).
 *
 * Everything outside `# === AI === ... # === END AI ===` is ignored.
 * On syntax/validation errors inside the block, preserves `lastValid` (if any)
 * and surfaces human-readable warnings rather than throwing.
 */
export function parseNotesDirectives(
    notesText: string | null | undefined,
    lastValid?: NotesDirectives | null,
): NotesParseResult {
    if (!notesText) {
        return {
            ok: true,
            hasBlock: false,
            directives: defaultNotesDirectives(),
            warnings: [],
            fallbackUsed: false,
        };
    }

    const lines = notesText.split(/\r?\n/);
    const startIdx = lines.findIndex((l) => START_MARKER.test(l.trim()));
    if (startIdx < 0) {
        return {
            ok: true,
            hasBlock: false,
            directives: defaultNotesDirectives(),
            warnings: [],
            fallbackUsed: false,
        };
    }

    let endIdx = lines.length;
    for (let i = startIdx + 1; i < lines.length; i++) {
        if (END_MARKER.test(lines[i].trim())) {
            endIdx = i;
            break;
        }
    }

    const warnings: string[] = [];
    const parsed: NotesDirectives = defaultNotesDirectives();
    let fatalError = false;

    for (let lineNo = startIdx + 1; lineNo < endIdx; lineNo++) {
        const cleaned = stripInlineComment(lines[lineNo]);
        if (!cleaned) continue;

        const colon = cleaned.indexOf(":");
        if (colon <= 0) {
            warnings.push(
                `Invalid AI Notes line ${lineNo + 1}: '${cleaned}' (expected 'key: value').`,
            );
            fatalError = true;
            continue;
        }

        const key = cleaned.slice(0, colon).trim().toLowerCase();
        const rawVal = cleaned.slice(colon + 1).trim();

        switch (key) {
            case "mode": {
                const v = unquote(rawVal).toLowerCase() as OperatingMode;
                if (!VALID_MODES.has(v)) {
                    warnings.push(
                        `Invalid mode '${rawVal}' in AI Notes (expected paused | advisor | approval | autonomous).`,
                    );
                    fatalError = true;
                } else {
                    parsed.mode = v;
                }
                break;
            }
            case "cash_reserve":
            case "cashreserve": {
                const n = Number(unquote(rawVal));
                if (!Number.isFinite(n) || n < 0) {
                    warnings.push(
                        `Invalid cash_reserve '${rawVal}' in AI Notes (expected non-negative number).`,
                    );
                    fatalError = true;
                } else {
                    parsed.cashReserve = Math.floor(n);
                }
                break;
            }
            case "specialist_reserve":
            case "specialistreserve": {
                const n = Number(unquote(rawVal));
                if (!Number.isFinite(n) || n < 0) {
                    warnings.push(
                        `Invalid specialist_reserve '${rawVal}' in AI Notes (expected non-negative number).`,
                    );
                    fatalError = true;
                } else {
                    parsed.specialistReserve = Math.floor(n);
                }
                break;
            }
            case "attack": {
                const list = parseStringList(rawVal);
                if (list == null) {
                    warnings.push(
                        `Invalid attack list '${rawVal}' in AI Notes (expected e.g. '[PlayerAlias]').`,
                    );
                    fatalError = true;
                } else {
                    parsed.attack = list;
                }
                break;
            }
            case "expect_tech":
            case "expecttech": {
                const list = parseStringList(rawVal);
                if (list == null) {
                    warnings.push(
                        `Invalid expect_tech list '${rawVal}' in AI Notes.`,
                    );
                    fatalError = true;
                } else {
                    const normalized: string[] = [];
                    for (const item of list) {
                        const t = item.toLowerCase();
                        if (!VALID_TECHS.has(t)) {
                            warnings.push(
                                `Unknown technology '${item}' in expect_tech.`,
                            );
                            fatalError = true;
                        } else {
                            normalized.push(t);
                        }
                    }
                    parsed.expectTech = normalized;
                }
                break;
            }
            case "neutral_border":
            case "neutralborder": {
                const v = unquote(rawVal).toLowerCase();
                let mapped: NeutralBorderPosture | null = null;
                if (v === "defend" || v === "deterrence") mapped = "defend";
                else if (v === "open" || v === "unfortified") mapped = "open";
                if (!mapped) {
                    warnings.push(
                        `Invalid neutral_border '${rawVal}' in AI Notes (expected defend | open).`,
                    );
                    fatalError = true;
                } else {
                    parsed.neutralBorder = mapped;
                }
                break;
            }
            default: {
                warnings.push(`Unknown directive key '${key}' in AI Notes.`);
                fatalError = true;
                break;
            }
        }
    }

    if (fatalError) {
        return {
            ok: false,
            hasBlock: true,
            directives: lastValid
                ? structuredClone(lastValid)
                : defaultNotesDirectives(),
            warnings,
            fallbackUsed: true,
        };
    }

    return {
        ok: true,
        hasBlock: true,
        directives: parsed,
        warnings,
        fallbackUsed: false,
    };
}
