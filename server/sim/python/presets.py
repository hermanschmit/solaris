"""Game presets the commander can train and play on.

The preset is chosen with the SOLARIS_SETTINGS environment variable (the
name of a file in server/config/game/settings/official), "standard" by
default. Everything that depends on the number of players or on the rules
comes from here, so the 10-player setup is unchanged unless you ask for
another preset:

    SOLARIS_SETTINGS=1v1 python3 ppo_train.py data/1v1-ft 150 156 13 - 200

players            seats in a game
seats              seats the commander being trained plays in each game
league_opponents   seats league members play in league games
ship_share_bonus   per-cycle reward for the seat's share of all ships
builtin            AiConfig overrides for every built-in AI seat (and the
                   built-in baseline in evaluations)
"""

import os

PRESETS = {
    "standard": dict(players=10, seats=3, league_opponents=4, ship_share_bonus=0.05,
                     builtin={}),
    # The official 1v1 preset: 80 stars, a 400-tick limit, weapons fixed at
    # 5, capital loss eliminates. Its free stars start beyond the built-in
    # AI's claim range, which it leaves only when random research raises
    # hyperspace or scanning; researchClaimRangeWhenStranded makes it do
    # so. In a two-player game every ship gained is one the opponent lost,
    # so the ship share bonus would mostly reward hoarding: it is off.
    "1v1": dict(players=2, seats=1, league_opponents=1, ship_share_bonus=0.0,
                builtin={"researchClaimRangeWhenStranded": True}),
}

NAME = os.environ.get("SOLARIS_SETTINGS", "standard")
if NAME not in PRESETS:
    raise ValueError(f"Unknown preset {NAME!r}; known: {', '.join(PRESETS)}")
PRESET = PRESETS[NAME]
PLAYERS = PRESET["players"]
SEATS = PRESET["seats"]


def builtin_configs(agents, ai_configs=None):
    """AiConfig overrides for every seat not in agents: the preset's
    built-in tuning, under any per-seat overrides given (personas)."""
    out = {}
    for seat in range(PLAYERS):
        if seat in agents:
            continue
        config = {**PRESET["builtin"], **(ai_configs or {}).get(seat, {})}
        if config:
            out[seat] = config
    return out
