// Jev-Omni's prompt (load_model.predict) and the Gemma 4 chat template for one user text turn with
// add_generation_prompt=True, enable_thinking=False. The template trims user content with Jinja's `trim`, which is
// Python's str.strip(): its whitespace set differs from String.prototype.trim (no U+FEFF, but U+001C-U+001F and
// U+0085 are included), so it is reproduced exactly.

export interface Question { state: string; question: string; options: string[] }

export const MIN_OPTIONS = 2;
export const MAX_OPTIONS = 256;

export function renderPrompt({ state, question, options }: Question): string {
  const choices = options.map((v, i) => `${i + 1}. ${v}`).join("\n");
  return `${state}\n\n---\n\nQUESTION: ${question}\n\nOPTIONS:\n${choices}\n\n`
    + `Reply with only the number of the correct option (1-${options.length}).\n`
    + "Output a single number and nothing else.";
}

const PY_WHITESPACE = new Set([
  0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x85, 0xa0, 0x1680,
  0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a,
  0x2028, 0x2029, 0x202f, 0x205f, 0x3000,
]);

export function pyStrip(s: string): string {
  let a = 0, b = s.length;
  while (a < b && PY_WHITESPACE.has(s.charCodeAt(a))) a++;
  while (b > a && PY_WHITESPACE.has(s.charCodeAt(b - 1))) b--;
  return s.slice(a, b);
}

/** The chat-templated text for one user turn, tokenized without adding special tokens (<bos> is in the text). */
export function chatText(userText: string): string {
  return `<bos><|turn>user\n${pyStrip(userText)}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>`;
}
