"""OpenEvolve target — pokemon_red milestone library (Stage S v2 baseline).

Loaded via the POKEMON_MILESTONE_LIBRARY_PATH override in
agents/pokemon_red/game_adapter.py. The override-loader does an
importlib spec_from_file_location import on this file (or any
OpenEvolve-mutated descendant) and reads the resulting module's
`_POKEMON_MILESTONE_LIBRARY` attribute.

Only the dict between EVOLVE-BLOCK markers is mutated by the
mutation LLM. The MilestoneSpec import is fixed scaffolding.

A valid candidate must:
- Define `_POKEMON_MILESTONE_LIBRARY: dict[int, MilestoneSpec]`.
- Cover at least one score gate in {5, 6, 7}.
- Use only `MilestoneSpec(name, description, suggested_tools,
  requires_location=...)` — no other constructors.
- Use only map-names that exist in the pokered map graph
  (PalletTown, Route1, ViridianCity, ViridianMart, OaksLab,
  RedsHouse1f, RedsHouse2f, ViridianPokecenter, etc.).
"""

from agents.macla.macla_lib import MilestoneSpec

# EVOLVE-BLOCK-START
_POKEMON_MILESTONE_LIBRARY: dict[int, MilestoneSpec] = {
    5: MilestoneSpec(
        name="EnterViridian",
        description=(
            "Walk into Viridian City (any Viridian-named map). Route 1 "
            "leads directly north from Pallet Town."
        ),
        suggested_tools=["move_to"],
        requires_location="ViridianCity",
    ),
    6: MilestoneSpec(
        name="GetOaksParcel",
        description=(
            "Pick up Oak's Parcel from the Viridian Mart — enter the Mart "
            "(blue-roofed building in Viridian City) and talk to the clerk "
            "at the counter."
        ),
        suggested_tools=["move_to", "interact_with_object", "continue_dialog", "a"],
        requires_location="ViridianMart",
    ),
    7: MilestoneSpec(
        name="DeliverOaksParcel",
        description=(
            "Return to Pallet Town and deliver Oak's Parcel to Professor "
            "Oak in his lab (south side of Pallet Town)."
        ),
        suggested_tools=["move_to", "interact_with_object", "continue_dialog", "a"],
        requires_location="OaksLab",
    ),
}
# EVOLVE-BLOCK-END
