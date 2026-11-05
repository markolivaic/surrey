// The page prints every number in one format: a share as a percentage, anything else with two
// decimals, a count whole. These run the page's own formatting over the figures it ships with and
// fail on any number printed with more than two decimals.
//
//   npm test

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { count, num, share, show } from "../src/format.js";
import { asprsHtml, stripHtml } from "../src/hero.js";
import { m1CardHtml } from "../src/m1card.js";

const read = (path) => readFileSync(new URL(path, import.meta.url), "utf-8");
const figures = JSON.parse(read("../public/figures.json"));
const asprs = JSON.parse(read("../public/asprs.json"));
const TOO_LONG = /\d\.\d{3,}/;

test("a share is a percentage, whole above 10% and to one decimal below", () => {
  assert.equal(share(0.9726), "97%");
  assert.equal(share(0.0312), "3.1%");
  assert.equal(share(0), "0%");
  assert.equal(share(0.0002), "under 0.1%");
  assert.equal(share(1), "100%");
});

test("anything else has two decimals, and a count is whole with separators", () => {
  assert.equal(num(-1.4439), "-1.44");
  assert.equal(num(0.3574), "0.36");
  assert.equal(num(121.77), "121.77");
  assert.equal(num(43938643), "43,938,643");
  assert.equal(count(23.0), "23");
});

test("every placeholder on the page names a format, and none prints more than two decimals", () => {
  const html = read("../index.html");
  const placeholders = [...html.matchAll(/<span[^>]*data-q="([^"]+)"[^>]*>/g)];
  assert.ok(placeholders.length > 50, `only ${placeholders.length} placeholders found`);
  for (const [tag, key] of placeholders) {
    const kind = tag.match(/data-format="([^"]+)"/)?.[1];
    assert.ok(kind, `${key} names no format`);
    assert.ok(figures[key], `${key} is not in figures.json`);
    const out = show(figures[key].value, kind);
    assert.doesNotMatch(out, TOO_LONG, `${key} prints as ${out}`);
  }
});

test("the M1 card, the first screen's table and its strip print no long decimals and no gaps", () => {
  const colourOf = () => "#000";
  for (const [name, out] of [["M1 card", m1CardHtml(figures)], ["table", asprsHtml(asprs, figures, colourOf)], ["strip", stripHtml(figures)]]) {
    const text = out.replace(/<[^>]+>/g, " ");
    assert.doesNotMatch(text, TOO_LONG, `${name}: ${text.match(TOO_LONG)?.[0]}`);
    assert.doesNotMatch(text, /missing|undefined|NaN/, `${name} has a figure it could not resolve`);
  }
});
