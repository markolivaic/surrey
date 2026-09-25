// Fine class ids (mbendjilali/DALES-2 order, then M1's water and bridge deck) to the coarse
// groups the page names. Colours are in docs/design/palette.json. The legend is built from the classes
// actually present in the loaded area, never from this table alone.

export const FINE = ["ground", "vegetation", "car", "powerline", "fence", "tree", "pickup", "van-truck",
  "heavy-duty", "utility-pole", "light-pole", "traffic-pole", "habitat", "complex", "annex",
  "water", "bridge_deck"];

export const GROUP_OF = {
  ground: "ground", vegetation: "vegetation", tree: "vegetation",
  habitat: "building", complex: "building", annex: "building",
  car: "vehicle", pickup: "vehicle", "van-truck": "vehicle", "heavy-duty": "vehicle",
  "utility-pole": "pole", "light-pole": "pole", "traffic-pole": "pole", powerline: "pole",
  fence: "fence", water: "water", bridge_deck: "bridge deck",
};

export function legendFor(classesPresent) {
  const groups = new Map();
  for (const [id, count] of Object.entries(classesPresent)) {
    const g = GROUP_OF[FINE[Number(id)]];
    groups.set(g, (groups.get(g) || 0) + count);
  }
  return [...groups.entries()].sort((a, b) => b[1] - a[1]).map(([group, points]) => ({ group, points }));
}
