"""Rotary position from sequence index and DOM path, on fixed plane ranges.

WHAT THIS IS
------------
nanoGPT has no rotary encoding. This module gives every attention head one, and
the *source* of the angle is the arm under test. A head of width ``hd`` has
``hd/2`` rotary planes (plane ``i`` pairs dims ``i`` and ``i + hd/2``), and the
planes are split ONCE, the same way in every arm:

  planes ``0 .. seq_planes-1``       SEQUENCE position
                                     angle = t * base^(-i / seq_planes) -- a full
                                     spectrum over those planes, as partial-rotary
                                     models (GPT-NeoX, GLM) do.
  planes ``seq_planes ..``           the TREE. Level ``k`` of the token's DOM path
                                     owns ``planes_per_level`` planes and rotates
                                     them by  present_k * theta[k, p] + rank_k * step[k, p].

Which ranges rotate is the arm:

  ``rope_seq``     sequence planes rotate, tree planes sit at angle 0
  ``rope_graph``   tree planes rotate, sequence planes sit at angle 0
  ``rope_both``    both rotate
  ``rope_leaf``    tree planes AND the token-leaf planes rotate, sequence at 0

and ``pos_mode='none'`` does not build this module at all (model.py). So the four
arms are a 2x2: each factor owns the same planes in every arm it appears in, and
an unrotated plane is a plain content-matching channel, not a broken one.

THE TREE STEPS
--------------
agon's ``TreeRopeConfig`` defaults, verbatim. The fast plane of a level steps
``sqrt(2 * (1 - 0.95))`` per sibling, which resolves adjacent siblings; the slow
plane steps 1/16 of that and counts to ~318 siblings before it wraps. ``theta``
is a per-level occupancy offset that starts at EXACTLY zero, so "rank 0 at this
level" and "not this deep" coincide at init and are separated by training. Both
are learned. Absent levels (``ABSENT``) contribute exactly zero on their planes.

``BASE_STEP`` is written as the expression, not the rounded 0.316: agon's own
validator rejects the rounded value.

THE TOKEN-LEAF LEVEL (cell 2, ``token_planes > 0``)
---------------------------------------------------
Under the tree levels every token of one text node shares a coordinate, so the
tree planes cannot order tokens within a node. The token-leaf level makes every
token its own leaf: the path gains one trailing column, the token's index
within its text node, and ``token_planes`` planes rotate by
``present * theta_tok[p] + within * step_tok[p]``. The steps are a ladder of
ratio 4 inside the level -- ``BASE_STEP / 4**p`` -- so with four planes the
level resolves adjacent tokens and counts to ~20 / 80 / 318 / 1270 tokens before
each plane wraps. Measured on the cell-1 pack: tokens per node mean 8.0, p50 4,
p90 14, p99 80; 0.02% of nodes exceed 318 tokens and hold 1.6% of tokens.
Only ``rope_leaf`` rotates these planes; ``rope_graph`` and ``rope_both``
leave them at angle 0 and ignore the trailing column.

THE SPLIT ARMS (2026-09-26): ``rope_head_split``, ``rope_layer_split``
----------------------------------------------------------------------
Every arm above splits ONE head's 64 planes into a 16-plane sequence band and a
44+4-plane tree band. That is the configuration the multi-axis literature argues
against: MRoPE-I asks for "Full Frequency Utilization" and HARoPE objects to
narrow contiguous bands that reuse the same spectrum. The split arms test the
alternative the field actually recommends -- give a factor a whole head, or a
whole layer, and let it span the FULL width.

  ``rope_head_split``   heads ``0 .. nh/2-1`` rotate by SEQUENCE over all 64
                        planes; heads ``nh/2 .. nh-1`` rotate by the TREE.
                        Needs per-head angles, hence the 4-D branch in
                        ``apply_rotary``. ``n_head`` must be even.
  ``rope_layer_split``  EVEN layers (0,2,4,...) rotate by SEQUENCE over all 64
                        planes; ODD layers (1,3,5,...) rotate by the TREE.
                        ``forward`` returns TWO ``(cos, sin)`` pairs and
                        ``model.py`` selects one per layer. This is the
                        sequence-vs-tree analogue of Llama 4's iRoPE, which
                        interleaves RoPE layers with position-free ones.

At full width each factor is allocated:

  SEQUENCE   all ``n_planes`` carry the spectrum, ``base ** (-i / n_planes)``.
             So the config must set ``rope_seq_planes == head_dim/2``, and the
             module REFUSES otherwise rather than silently reinterpreting it.
  TREE       ``levels * planes_per_level`` level planes then ``token_planes``
             token planes, both exactly as ``rope_leaf`` computes them, laid out
             from plane 0. Any REMAINING planes stay at angle 0 -- unrotated
             content channels, which is what partial-RoPE models do. With 22
             levels x 2 + 4 token = 48 of 64, 16 planes are left unrotated.
             Placement inside the head is arbitrary for the tree because its
             rates come from the learned ladder, not from the plane index.

Both split arms hold tree AND token parameters and the sequence buffer, so they
are not a subset of any earlier arm; they reallocate rather than remove.

THE PER-LAYER PLAN (2026-09-26): ``rope_layer_plan``
----------------------------------------------------
``rope_layer_split`` hardcodes "even layers sequence, odd layers tree". That is one
point in a space worth sweeping, so ``rope_layer_plan`` takes the assignment as a
string of length ``n_layer``, one character per layer in order:

  ``s``   that layer rotates by SEQUENCE at full width (all ``n_planes``)
  ``t``   that layer rotates by the TREE at full width (levels then token planes)
  ``n``   that layer gets NO rotation at all -- ``apply_rotary`` is skipped, not
          called with cos=1/sin=0, so it costs nothing

So ``ststst`` reproduces ``rope_layer_split`` on 6 layers, ``sssttt`` is
sequence-then-tree, ``tttsss`` is tree-then-sequence, and ``ssttnn`` is thirds with
a position-free block. The ``n`` character makes this a strict generalisation of
Llama 4's iRoPE / Cohere's RNoPE-SWA, which interleave rotating layers with
position-free ones; here the rotating layers can carry two different coordinates.

Only the factors the plan actually names are built: a plan with no ``t`` allocates
no tree parameters, and a plan with no ``s`` needs no sequence spectrum. A plan of
all ``n`` is refused -- that is ``pos_mode='none'`` and is almost certainly a typo.

THE SYMBOL SLOT (2026-09-28): ``rope_ref_slot``
-----------------------------------------------
Every arm above rotates by where a token *is*. This one adds a group that rotates
by what it *refers to*, which is the first coordinate here that is not a property
of the token alone.

A rotary score depends on the DIFFERENCE of two angles, so two tokens sharing an
angle in some group are coincident there. Give every mention of a symbol the same
slot -- the rank, in first-mention order within the record, of the symbol it
belongs to -- and all mentions of one symbol become mutually coincident in the
slot planes while different symbols do not. **"Same thing" becomes "angularly
coincident."** This is CO-REFERENCE, not pointing: nobody names a target, they
simply agree, so one group suffices and no matching axis is needed.

  ``rope_ref_slot``   sequence + tree + the SLOT group. Allocation is
                      16 seq + 22 levels x 2 + ``slot_planes`` + remainder
                      unrotated. The 2 slot planes come out of the 4 token-leaf
                      planes, measured worthless when sequence rotates
                      (``rope_all`` vs ``rope_both``: +0.0001, t=+0.12, n=5).

The slot arrives as ``pos['slot']``, a SEPARATE key from ``pos['path']``, so the
path keeps exactly the ``levels (+1)`` columns every earlier arm expects and no
existing arm can see this column. ``ABSENT`` contributes exactly zero, as it does
for an absent tree level.

Slots are built by ``convert_treepos.py --slots`` from the sidecar's ``node_ref``
edges, NOT from its ``node_ident`` angles: measured over 986,360 reference pairs a
use and its own binder share an identity value only 70.43% of the time, because
the hash covers the node TAG and a binder's tag differs from a use's. See
``build_tok_slot`` for that measurement and for the causality gate.

HONEST LIMITATION. A slot is a categorical code wearing a continuous coordinate's
clothing: slot 5 and slot 6 are consecutive but not related, so slot DISTANCE
should carry nothing and only coincidence should matter. That is testable -- if
the learned ``step_slot`` moves toward separating adjacent slots maximally rather
than toward a smooth ladder, the model is telling us it uses the axis
categorically. The tree planes already showed the model discarding our
hand-designed ladder, so we know how to read this.

THE GRID POINTER (R7, 2026-09-28): ``rope_ref_path``
----------------------------------------------------
``rope_ref_slot`` points by agreement: mentions of one symbol share a slot and
become coincident. That works where a name has no location. In a SPREADSHEET the
name IS the location -- ``A1`` means row 1, column A, readable from the reference
text itself -- so there is no need to intern anything. This arm rotates by the
referent's ACTUAL ``(row, column)``.

THE PROPERTY THAT MAKES IT WORK, AND THE TRAP IT AVOIDS. Rotary position only
ever compares THE SAME PLANE INDEX between query and key. So a reference's
coordinate is useless in a group of its own: the query would rotate planes the
key never touches, and the two would never interact. The group has to be written
by BOTH sides.

It is, and by construction rather than by arrangement. ONE group of
``ref_planes``, and EVERY token writes a ``(row, column)`` into it:

  * a token that carries a reference (``ref_row >= 0``) writes THE TARGET's
    coordinate, from ``tok_ref_row`` / ``tok_ref_col``;
  * every other token writes ITS OWN enclosing cell's coordinate, read out of the
    path at levels ``REF_ROW_LEVEL`` and ``REF_COL_LEVEL``.

Both sides then run the same arithmetic through the same ``step_ref`` /
``theta_ref``, so a reference's address tokens land on BIT-EXACTLY the angle of
the cell they name. That is the coincidence, and it cannot drift apart without
deleting one of the two branches.

The path levels are the right source because the sheets spine is
``root -> sheet -> row -> cell``, so a token's enclosing row and column sit at
path levels 1 and 2 **whatever its own depth** -- a literal value at depth 4 and a
``Ref`` operand six levels down both report the same enclosing cell. And the
builder produces ``tok_ref_row`` / ``tok_ref_col`` from the same ``coord_of()``
that writes ``node_path``, so the two are already in one space; that invariant is
asserted on the data side by
``test_sheet_pack.py::test_reference_coordinate_equals_its_targets_own_path``.

NO ABSENT CLASS FOR THE COMMON CASE, which is the lesson of ``rope_ref_slot``.
There, 57% of tokens had no slot, all mutually coincident, and the model spent the
group's ``theta`` on a crude presence flag instead of on identity: measured,
``theta_slot`` went from exactly 0 to 1.23 rad while ``step_slot`` -- which
discriminates *which* symbol -- barely moved. Here every token inside a cell has a
real coordinate, so there is nothing for a presence flag to buy. ``theta_ref`` is
still present and still starts at exactly zero, but note what it can do: it is
added with ``present = 1`` for every token that has a coordinate, so among those it
is a CONSTANT and cancels in every query-key difference. It can only separate
in-cell tokens from tokens with no cell at all.

Tokens with no cell coordinate contribute EXACTLY zero on these planes, as an
absent tree level does. On the sheets packs that is the framing ``EOT``, the
tab/newline separators, and the trailing defined-names section.

A whole-row reference target has ``ref_col = ABSENT`` (384 of 24,477 targets). The
selector is keyed on the ROW, so such a token takes the target's row and the
target's absent column rather than mixing in its own column -- the group always
holds one coordinate pair from one source.

HONEST LIMITATION: the defined-names section is misread. Its nodes hang off a
``Names`` table that is also a depth-1 child of the root, so a defined name sits at
the same path level a ROW does and its target one below, where a COLUMN does. Those
tokens therefore report a spurious cell coordinate rather than none. Fixing it
properly needs a marker from the builder; the arm is shipped with the fraction
measured and recorded in ``config/sheet_ref.py`` instead.

THE ANCHOR (R7b, 2026-09-28): ``rope_ref_anchor`` -- and why R7 lost
--------------------------------------------------------------------
``rope_ref_path`` MEASURED WORSE than its baseline: +0.0075 nats/token against
``rope_both``, worse at 20 of 20 milestones, seed 0. Its learned parameters exclude
the ``rope_ref_slot`` failure above -- ``step_ref`` moved 3-6x on the slow planes, so
the coordinate was genuinely used, not abandoned for a presence flag. The design was
the problem, in two ways:

  1. **The coordinate had no sheet.** ``REF_PATH_AXES`` is 2. A reference crossing to
     another sheet therefore coincided with whatever cell held those ``(row, col)``
     on the reference's OWN sheet. Measured on 1,500 workbooks: **17.25% of
     references are cross-sheet**, and 25.5% of workbooks hold several sheets. Those
     references did not merely fail to help; they asserted a wrong match confidently.
  2. **The side effect was twice the signal.** A token's own cell coordinate is
     already in the tree planes at levels 1 and 2, so for the 63% of tokens carrying
     no reference the pointer group DUPLICATED it. Two tokens in one cell then agreed
     on 8 planes -- 4 tree plus 4 pointer -- while a reference and its target agreed
     on only the 4 pointer planes and actively DISagreed on the tree ones. The
     mechanism was half the strength of an artifact of its own construction.

So this arm has no pointer group at all. A token that carries a reference has its
own ``REF_ANCHOR_LEVELS`` -- sheet, row, column -- **OVERWRITTEN** by its target's,
and in every other respect the arm is ``rope_both``:

  * a reference and its target now agree on exactly the planes, and by exactly the
    amount, that two tokens in one cell do. No doubling, nothing to swamp.
  * nothing extra is written for the tokens that carry no reference, so there is no
    duplicate coordinate and no new all-zero class to be repurposed as a flag.
  * it costs **zero extra planes**. The allocation is the ordinary
    ``seq + levels * ppl + token_planes`` and the arm rotates precisely the planes
    ``rope_both`` rotates, so the paired comparison carries no plane-count confound.
    ``rope_ref_path`` had to disclose 64 against 60.

Both sides still write the same planes, which is the property that made R7 correct
and is what a private group destroys -- but here it is automatic rather than
arranged: the target writes its coordinate into those levels simply by being where
it is. A test pins it anyway.

The levels BELOW the column are deliberately untouched. They hold the token's
position *within the formula* -- which operand it is -- which is still true of a
reference; only its answer to "which cell is this about" changes. A reference
therefore keeps its role in the expression and loses its own address, which for a
formula operand is the right trade: what matters about ``A50`` is what it names.

A reference whose target is a defined name, or is otherwise not a cell, reports
ABSENT on all three axes. The selector keys on the ROW, so such a token keeps its own
coordinates rather than being half-overwritten.

THE OFFSET (R11b, 2026-09-29): ``rope_ref_delta`` -- what the anchor still cannot say
-------------------------------------------------------------------------------------
The anchor makes ``angle(reference) - angle(target) == 0``. But **zero is also the
difference between two tokens in the same cell**, so no attention head can
distinguish "the cell I point at" from "my own neighbours" -- they are the same
geometric state. R7b fixed the MAGNITUDE of R7's swamping problem and left the
CONFUSABILITY untouched.

This arm is the anchor plus ONE learned constant, added to the referencing token's
anchor-level angles only:

    angle(reference) = angle_from(target's coordinate) + delta_ref
    angle(target)    = unchanged

so the difference is exactly ``delta_ref`` -- a referent sits at ``delta``, a
neighbour at ``0``, and they are finally different states. Two more properties come
free from the group structure, neither of which coincidence can express at all:

  * **Direction.** use->def is ``+delta``, def->use is ``-delta``. Coincidence is
    symmetric by construction and cannot tell "I point at you" from the reverse.
  * **Composition.** A two-hop chain is ``2*delta``. Nothing represents transitivity;
    the algebra supplies it.

ONE-SIDED IS THE WHOLE THING. Applied to both sides the constant cancels in every
query-key difference and the arm reduces to ``rope_ref_anchor`` exactly -- it would
train, converge and mean nothing, the same shape of silent failure as giving
references a private plane group. ``_add_ref_delta`` is one-sided by construction
because the target is not a referencing token, and a test pins it.

``delta_ref`` starts at EXACTLY zero, so at step 0 this arm is bit-identical to
``rope_ref_anchor``: strictly more expressive, able only to differentiate. That makes
the diagnostic the result -- if training moves delta off zero, the model wanted
"referent" to be a different state from "co-located", which is the argument above
confirmed directly. Read it with ``read_ref_delta()``.

This is the REFERENCE half of R11. ``rope_rel`` is the hierarchy half: it implements
the same constant-offset idea for the PARENT/CHILD relation, as ``delta * depth``.
docs/design-reference-rotation.md specifies both and only the second was built.

INITIALISATION USES NO RANDOMNESS, so building this module never moves the
block stack's init -- the transformer is bit-identical across the four arms at a
seed, which is what makes an arm comparison a comparison of position sources
and not of two random models.
"""

import math

import torch
import torch.nn as nn

#: ``rope_all`` (2026-09-26) is the arm that was missing: sequence AND the tree AND the
#: within-node token index, all rotating. ``rope_both`` is only sequence + tree -- it never
#: had the token-leaf planes -- so ``both`` vs ``leaf`` differs on TWO axes at once and
#: neither is a superset of the other. ``rope_all`` is the superset of every other arm.
#:
#: The SPLIT arms (2026-09-26) ask a different question: instead of carving one head's 64
#: planes into a sequence band and a tree band, give each factor a WHOLE head (or a whole
#: layer) at FULL WIDTH. See ``SPLIT_MODES`` below.
SPLIT_MODES = ("rope_head_split", "rope_layer_split")

#: ``rope_layer_plan`` (2026-09-26) generalises ``rope_layer_split`` from "even/odd" to an
#: arbitrary per-layer assignment given as a string over ``PLAN_CHARS``. See the module
#: docstring. It is deliberately NOT in ``SPLIT_MODES``: the split arms always build every
#: factor, whereas a plan builds only the factors its own string names.
PLAN_MODE = "rope_layer_plan"

#: One character per layer: sequence at full width, tree at full width, or no rotation.
PLAN_CHARS = ("s", "t", "n")

#: ``rope_ref_slot`` (2026-09-28) is the first arm to rotate by what a token *refers to*
#: rather than only by where it *is*. See the SYMBOL SLOT section of the module docstring.
REF_MODE = "rope_ref_slot"

#: ``rope_rel`` (R11a, 2026-09-28) rotates by a DIRECTED RELATION instead of by presence.
#: Its planes carry ``depth * step_rel``, so a parent and its child differ by EXACTLY
#: ``step_rel`` and a grandparent by ``2 * step_rel`` -- direction (+/-), relation type (a
#: different constant per relation) and composition (n hops = n deltas) all fall out of the
#: group structure, none of which coincidence can express.
#:
#: Two properties distinguish it from the slot group:
#:   * NO ``theta``/presence term. Every token has a depth, so unlike a symbol slot there is
#:     no absent class and nothing to separate. A token outside the tree entirely has every
#:     level ``ABSENT``, so its depth counts to 0, which is distinct from a depth-1 child of
#:     the root. That is deliberate and is the documented value for out-of-tree tokens.
#:   * It is ADDITIONAL to the per-level rank planes, never a replacement. ``depth * delta``
#:     alone would flatten every node at one depth onto the same angle and destroy sibling
#:     identity; the rank planes keep it. See docs/design-reference-rotation.md, "The
#:     collapse this must be designed around".
#:
#: Prior art to cite and distinguish: RotatE (Sun et al., ICLR 2019, arXiv 1902.10197)
#: defines each relation as a rotation from source to target and proves symmetry /
#: antisymmetry / inversion / composition -- but as a knowledge-graph link-prediction
#: embedding, not as a positional encoding inside a transformer. Our claim is the placement
#: and the partitioning, never the relational rotation itself.
REL_MODE = "rope_rel"

#: ``rope_ref_anchor`` (R7b, 2026-09-28) is the SUCCESSOR to ``rope_ref_path``, built because
#: that arm measured WORSE than its baseline: +0.0075 nats/token against ``rope_both``, worse at
#: 20 of 20 milestones, seed 0. Its learned parameters rule out the old presence-flag failure --
#: ``step_ref`` moved 3-6x on the slow planes, so the coordinate was genuinely used. Two design
#: faults explain the loss, and this arm removes both:
#:
#:   1. NO SHEET. ``REF_PATH_AXES`` is 2, so a reference crossing to another sheet coincided
#:      with whatever cell held those coordinates on its OWN sheet -- 17.25% of references,
#:      measured. A confident wrong answer, not a missing one.
#:   2. THE SIDE EFFECT WAS TWICE THE SIGNAL. A token's own cell coordinate already lives in the
#:      tree planes, so for the 63% of tokens with no reference the pointer group DUPLICATED it.
#:      Two tokens in one cell then matched on 8 planes (4 tree + 4 pointer) while a reference
#:      and its target matched on 4 and actively MISmatched on the tree planes.
#:
#: So there is no separate group at all. A token carrying a reference has its own sheet/row/
#: column levels OVERWRITTEN by its target's, and everything else is ``rope_both`` unchanged.
#: A reference and its target then match on exactly the same planes, by the same amount, as two
#: tokens in one cell -- no doubling, and the pointer is no longer the weaker of the two. It
#: also costs ZERO extra planes, so the allocation is byte-identical to ``rope_both`` and the
#: paired comparison has no plane-count confound, which ``rope_ref_path`` had to disclose.
REF_ANCHOR_MODE = "rope_ref_anchor"

#: ``rope_ref_path`` (R7, 2026-09-28) rotates by the referent's ACTUAL ``(row, column)``.
#: See the GRID POINTER section of the module docstring. The one thing to hold onto: there is
#: a SINGLE group and every token writes into it -- its target's coordinate if it carries a
#: reference, its own enclosing cell's otherwise -- because rotary position only ever compares
#: the same plane index between query and key, so a private group for references would encode
#: nothing the target could answer.
REF_PATH_MODE = "rope_ref_path"

#: ``rope_ref_delta`` (R11b, 2026-09-29) is ``rope_ref_anchor`` plus ONE degree of freedom: a
#: learned constant ``delta_ref`` added to the REFERENCING token's angle on the anchor levels,
#: so ``angle(reference) - angle(target) == delta_ref`` exactly rather than zero.
#:
#: WHY. The anchor arm makes the difference ZERO -- but zero is also the difference between two
#: tokens in the SAME cell, so no head can distinguish "the cell I point at" from "my own
#: neighbours": they are the same geometric state. R7b fixed the MAGNITUDE of the swamping
#: problem (no more doubling) and left the CONFUSABILITY. A non-zero constant separates them.
#:
#: It also recovers two properties coincidence cannot have at all, both free from the group
#: structure: DIRECTION (use->def is ``+delta``, def->use is ``-delta``; coincidence is
#: symmetric by construction) and COMPOSITION (a two-hop chain is ``2*delta``, so nothing has
#: to represent transitivity). This is the reference half of the R11 idea -- see
#: docs/design-reference-rotation.md, which specifies it and which ``rope_rel`` did NOT build:
#: that arm implements the same constant-offset idea for the PARENT/CHILD relation via
#: ``delta * depth``.
#:
#: ``delta_ref`` initialises to EXACTLY zero, so at step 0 this arm is bit-identical to
#: ``rope_ref_anchor``. It is therefore strictly more expressive and can only differentiate,
#: and whether delta moves off zero IS the result: it says directly whether the model wanted
#: "referent" to be a different state from "co-located".
REF_DELTA_MODE = "rope_ref_delta"

#: ``rope_depth`` (2026-09-29, author's proposal) is sequence position plus the token's tree DEPTH
#: and NOTHING ELSE. No per-level path planes at all: 16 sequence + 2 depth = 18 of 64.
#:
#: WHY IT MIGHT BE ENOUGH. A tree serialised in PREFIX (DFS pre-order) order is fully determined
#: by ``(emission order, depth)`` -- the parent of any node is the nearest PRECEDING node one
#: level shallower. So depth is a LOSSLESS encoding of the tree even though the tree is never
#: made explicit. Verified on the real Python sidecar: reconstructing each node's parent from
#: order and depth alone recovered 558,271 parents with ZERO errors over 4,000 records
#: (``test_depth_and_order_reconstruct_the_tree`` pins it). All three renders in this project are
#: prefix serialisations -- Python ``astcompact`` is a "compact prefix serialisation", the DOM is
#: document order, the sheets render walks the spine in reading order.
#:
#: WHY IT MIGHT NOT BE. Lossless in principle is not the same as usable by attention. The full
#: path gives DIRECT coincidence: two tokens in one subtree share a path prefix and are aligned
#: on those planes in one step. With depth, the relative rotation between two tokens is the depth
#: DIFFERENCE, and "the nearest preceding node at depth-1" is not a fixed offset -- attention has
#: to combine a sequence rotation with a depth rotation and do something argmax-shaped. It may
#: well not recover what the path hands it directly. That is the question the arm asks.
#:
#: TWO COMPARISONS, ANSWERING DIFFERENT QUESTIONS -- see config/py_depth.py, which says which is
#: which, because a later reader will quote whichever suits them:
#:   * vs ``rope_seq`` (16 planes): plane-adjacent and CLEAN. "Does depth add anything at all?"
#:   * vs ``rope_both`` (60 planes): carries an 18-vs-60 plane-count difference. "Does 2 match
#:     44?" If they measure the same, the result is AMBIGUOUS between "depth is sufficient" and
#:     "most of those 44 planes were doing nothing".
#:
#: It shares ``step_rel`` and ``_rel_angles`` with ``rope_rel`` rather than growing a second
#: depth path; the difference is only that ``rope_rel`` keeps the per-level planes and this arm
#: drops them. Same no-theta argument: every token has a depth, a token outside the tree has
#: depth 0, which is a real value distinct from a depth-1 child of the root, not a sentinel.
DEPTH_MODE = "rope_depth"

ROPE_MODES = (
    ("rope_seq", "rope_graph", "rope_both", "rope_leaf", "rope_all")
    + SPLIT_MODES
    + (PLAN_MODE, REF_MODE, REL_MODE, REF_PATH_MODE, REF_ANCHOR_MODE, REF_DELTA_MODE)
    + (DEPTH_MODE,)
)

#: Rope modes whose angles READ THE DOM PATH, and therefore need the ``_dompath`` sidecar
#: loaded by ``topos_data.PackedWindows(sidecar='path')``. ``rope_layer_plan`` is absent
#: because it depends on the plan string -- use ``plan_needs_path()``.
#: Kept here so a new arm cannot be added without the loader noticing (2026-09-26: the two
#: split arms were missing from train.py's hardcoded map and would have crashed on launch).
TREE_MODES = (
    ("rope_graph", "rope_both", "rope_leaf", "rope_all")
    + SPLIT_MODES
    + (REF_MODE, REL_MODE, REF_PATH_MODE, REF_ANCHOR_MODE, REF_DELTA_MODE)
    + (DEPTH_MODE,)
)

#: Rope modes that additionally need ``tok_slot.npy`` -- the per-token symbol slot, which
#: only a ``--slots`` conversion emits. Kept beside TREE_MODES for the same reason: the
#: loader derives its requirements from here instead of a hardcoded map, because a
#: hardcoded map already drifted once and would have killed two launches (commit 3ca636a).
SLOT_MODES = (REF_MODE,)

#: Rope modes that additionally need ``tok_ref_row.npy`` / ``tok_ref_col.npy`` -- the per-token
#: GRID POINTER, which only ``build_sheet_pack.py`` emits. Same reason this lives here rather
#: than in a map inside train.py: a hardcoded map has already drifted twice and killed a
#: launch each time (3ca636a, d132e3c).
REF_PATH_MODES = (REF_PATH_MODE,)

#: Rope modes that OVERWRITE their own path levels with a reference target's, rather than
#: carrying a separate pointer group. Needs all THREE pointer axes including the sheet.
#: ``rope_ref_delta`` belongs here because it overwrites identically -- it only adds a constant
#: afterwards -- so it has exactly the same sidecar requirement and the loader needs no new key.
REF_ANCHOR_MODES = (REF_ANCHOR_MODE, REF_DELTA_MODE)

#: The subset that additionally adds the learned constant offset. Separate from
#: ``REF_ANCHOR_MODES`` because that tuple answers "what does the loader owe this arm" while
#: this one answers "does it hold a delta_ref parameter".
REF_DELTA_MODES = (REF_DELTA_MODE,)

#: Every mode that reads the per-token grid pointer at all, in whatever shape. The loader
#: keys on this; only ``REF_ANCHOR_MODES`` additionally REQUIRES tok_ref_sheet.npy.
REF_COORD_MODES = REF_PATH_MODES + REF_ANCHOR_MODES

#: Which path levels carry the enclosing cell's row and column. The sheets spine is
#: ``root -> sheet -> row -> cell``, and a path holds the coordinates from the child of the
#: root downward, so level 0 is the sheet, 1 the row and 2 the column -- for EVERY token
#: inside a cell regardless of its own depth, which is what makes a deep formula operand
#: report the same enclosing cell as the literal beside it.
REF_SHEET_LEVEL, REF_ROW_LEVEL, REF_COL_LEVEL = 0, 1, 2

#: The anchor arm overwrites exactly these path levels, in this order, from a (sheet, row,
#: column) target. Levels BELOW the column are left untouched: they are the token's position
#: WITHIN the formula -- which operand it is -- and that is still true of a reference. Only
#: its answer to "which cell is this about" changes.
REF_ANCHOR_LEVELS = (REF_SHEET_LEVEL, REF_ROW_LEVEL, REF_COL_LEVEL)
REF_ANCHOR_AXES = len(REF_ANCHOR_LEVELS)
#: Index of the row within a ``(sheet, row, column)`` pointer. NOT the selector: a pointer
#: applies only when EVERY axis is present, because a partial overwrite is a malformed path.
#: Keying on the row alone -- which ``rope_ref_path`` did -- protected 0 token positions and
#: discarded 16, measured over the whole shipped pack.
REF_ANCHOR_ROW_AXIS = REF_ANCHOR_LEVELS.index(REF_ROW_LEVEL)
#: The overwrite is a single contiguous slice from level 0, which is what lets it be one
#: ``torch.where`` over ``tree_path[..., :REF_ANCHOR_AXES]``. Asserted rather than assumed: if
#: the sheets spine ever gains a level above the sheet, this fires instead of silently
#: overwriting the wrong three levels.
assert REF_ANCHOR_LEVELS == tuple(range(REF_ANCHOR_AXES)), (
    f"REF_ANCHOR_LEVELS must be the contiguous prefix 0..{REF_ANCHOR_AXES - 1}, "
    f"got {REF_ANCHOR_LEVELS}"
)


def plan_needs_path(layer_plan: str) -> bool:
    """True when a ``rope_layer_plan`` string has any tree layer, so needs the path sidecar."""
    return "t" in str(layer_plan or "").strip().lower()

#: Sentinel in a path for "the node is not this deep" / "token outside the tree".
ABSENT = -1

#: Fast-plane step per sibling: adjacent siblings resolvable at the 5% alias threshold.
BASE_STEP = math.sqrt(2 * (1 - 0.95))

#: Geometric ratio between a level's fast and slow plane.
LADDER_RATIO = 16.0

#: Geometric ratio between successive planes of the token-leaf level.
TOKEN_LADDER_RATIO = 4.0

#: Geometric ratio between successive planes of the symbol-slot level. Same ladder shape as
#: the tree levels: the fast plane resolves adjacent slots, the slow one counts to ~318
#: before it wraps. Measured slots per record are median 8 / max 75, so the slow plane alone
#: covers every record in the pack four times over.
SLOT_LADDER_RATIO = 16.0

#: Geometric ratio between successive planes of the RELATIONAL (depth) group. Same ladder
#: shape as a tree level: the fast plane resolves adjacent depths -- which is the whole
#: point, since a parent/child hop is a step of one -- and the slow plane counts to ~318
#: depths before it wraps, far past the deepest observed tree (Python 57, sheets 60).
REL_LADDER_RATIO = 16.0

#: Geometric ratio between successive planes of each GRID-POINTER axis. Same ladder shape as a
#: tree level: with 2 planes per axis the fast plane resolves adjacent rows (or columns) and
#: the slow one counts to ~318 before it wraps. Two coordinates alias only by coinciding on
#: BOTH planes, whose periods are ~19.9 and ~317.4, so the first genuine collision is ~6,300
#: rows apart -- past the p999 absolute row of 2,526 measured on the sheets corpus. Columns
#: reach p99 152, comfortably inside the slow plane alone.
REF_PATH_LADDER_RATIO = 16.0

#: The grid pointer is a (row, column) PAIR, so its planes are split evenly between two axes.
REF_PATH_AXES = 2


class GraphRoPE(nn.Module):
    """``(cos, sin)`` of shape ``(B, t, hd/2)`` for one forward, shared by every layer."""

    def __init__(
        self,
        head_dim: int,
        mode: str,
        seq_planes: int = 16,
        levels: int = 24,
        planes_per_level: int = 2,
        base: float = 10000.0,
        token_planes: int = 0,
        n_head: int = 0,
        layer_plan: str = "",
        n_layer: int = 0,
        slot_planes: int = 0,
        rel_planes: int = 0,
        ref_planes: int = 0,
    ) -> None:
        super().__init__()
        if mode not in ROPE_MODES:
            raise ValueError(f"mode {mode!r} is not one of {ROPE_MODES}")
        if head_dim % 2:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        n_planes = head_dim // 2
        self.is_split = mode in SPLIT_MODES
        self.is_plan = mode == PLAN_MODE
        self.is_ref = mode == REF_MODE
        self.is_rel = mode == REL_MODE
        self.is_ref_path = mode == REF_PATH_MODE
        self.is_ref_anchor = mode == REF_ANCHOR_MODE
        self.is_ref_delta = mode == REF_DELTA_MODE
        self.is_depth = mode == DEPTH_MODE
        # The delta arm overwrites exactly as the anchor does, so every anchor-side constraint
        # and code path is shared; the only difference is the constant added afterwards.
        self.anchors_path = self.is_ref_anchor or self.is_ref_delta
        if self.anchors_path and levels <= REF_COL_LEVEL:
            raise ValueError(
                f"{mode} overwrites path levels {REF_ANCHOR_LEVELS} with a "
                f"reference target's coordinate, so it needs rope_levels > {REF_COL_LEVEL}, "
                f"got {levels}"
            )
        if slot_planes and not self.is_ref:
            raise ValueError(
                f"rope_slot_planes={slot_planes} is set but mode is {mode!r}; only {REF_MODE} "
                "reads the symbol slot, so this allocation would be silently ignored"
            )
        if rel_planes and not (self.is_rel or self.is_depth):
            raise ValueError(
                f"rope_rel_planes={rel_planes} is set but mode is {mode!r}; only {REL_MODE} "
                f"and {DEPTH_MODE} read the relational depth, so this allocation would be "
                "silently ignored"
            )
        if ref_planes and not self.is_ref_path:
            raise ValueError(
                f"rope_ref_planes={ref_planes} is set but mode is {mode!r}; only "
                f"{REF_PATH_MODE} reads the grid pointer, so this allocation would be "
                "silently ignored"
            )
        self.plan = ()
        if self.is_plan:
            plan = tuple(str(layer_plan or "").strip().lower())
            if not plan:
                raise ValueError(
                    f"{PLAN_MODE} needs rope_layer_plan set, e.g. 'ststst' for 6 layers "
                    f"(one character per layer from {PLAN_CHARS})"
                )
            if n_layer <= 0:
                raise ValueError(f"{PLAN_MODE} needs n_layer > 0 to check the plan length")
            if len(plan) != n_layer:
                raise ValueError(
                    f"rope_layer_plan {''.join(plan)!r} has {len(plan)} entries but n_layer "
                    f"is {n_layer}; one character per layer is required"
                )
            bad = sorted({c for c in plan if c not in PLAN_CHARS})
            if bad:
                raise ValueError(
                    f"rope_layer_plan {''.join(plan)!r} contains {bad}, outside {PLAN_CHARS}"
                )
            if all(c == "n" for c in plan):
                raise ValueError(
                    f"rope_layer_plan {''.join(plan)!r} rotates nothing at all; that is "
                    "pos_mode='none', so this is almost certainly a config error"
                )
            self.plan = plan
        # FULL WIDTH per factor, for the split arms and for a plan alike. A plan only has to
        # satisfy the constraint for the factors its own string actually names.
        if self.is_split or self.is_plan:
            want_seq = self.is_split or ("s" in self.plan)
            want_tree = self.is_split or ("t" in self.plan)
            if want_seq and seq_planes != n_planes:
                # The sequence spectrum spans every plane, so the config must say so
                # explicitly rather than have 16 silently reinterpreted.
                raise ValueError(
                    f"{mode} gives the sequence factor the FULL head, so rope_seq_planes "
                    f"must equal head_dim/2 = {n_planes}, got {seq_planes}"
                )
            if want_tree:
                tree_used = levels * planes_per_level + token_planes
                if tree_used > n_planes:
                    raise ValueError(
                        f"{mode} tree allocation {levels} levels x {planes_per_level} + "
                        f"{token_planes} token planes = {tree_used} exceeds head_dim/2 = {n_planes}"
                    )
                if token_planes == 0:
                    raise ValueError(f"{mode} needs token_planes > 0 (every token its own leaf)")
                self.tree_used = tree_used
            if mode == "rope_head_split":
                if n_head <= 0:
                    raise ValueError("rope_head_split needs n_head > 0 to split the heads")
                if n_head % 2:
                    raise ValueError(f"rope_head_split needs an even n_head, got {n_head}")
        if self.is_ref:
            if slot_planes <= 0:
                raise ValueError(
                    f"{REF_MODE} needs rope_slot_planes > 0 -- that is the whole arm. Take them "
                    "from the token-leaf planes, which are worthless when sequence rotates."
                )
            used = seq_planes + levels * planes_per_level + token_planes + slot_planes
            if used > n_planes:
                raise ValueError(
                    f"{REF_MODE} allocation seq={seq_planes} + {levels} levels x "
                    f"{planes_per_level} + {token_planes} token + {slot_planes} slot = {used} "
                    f"exceeds head_dim/2 = {n_planes}"
                )
            self.ref_used = used
        if self.is_rel:
            if rel_planes <= 0:
                raise ValueError(
                    f"{REL_MODE} needs rope_rel_planes > 0 -- that is the whole arm. Take them "
                    "from the token-leaf planes, which are worthless when sequence rotates."
                )
            if token_planes:
                raise ValueError(
                    f"{REL_MODE} isolates the relational axis so that rope_both is the paired "
                    f"baseline, so it takes no token-leaf planes; got token_planes={token_planes}"
                )
            used = seq_planes + levels * planes_per_level + rel_planes
            if used > n_planes:
                raise ValueError(
                    f"{REL_MODE} allocation seq={seq_planes} + {levels} levels x "
                    f"{planes_per_level} + {rel_planes} relational = {used} exceeds "
                    f"head_dim/2 = {n_planes}"
                )
            self.rel_used = used
        if self.is_depth:
            if rel_planes <= 0:
                raise ValueError(
                    f"{DEPTH_MODE} needs rope_rel_planes > 0 -- the depth group is the whole "
                    "structural budget of this arm."
                )
            if token_planes:
                raise ValueError(
                    f"{DEPTH_MODE} spends its budget on depth alone so that rope_seq is the "
                    f"plane-adjacent baseline; it takes no token-leaf planes, got "
                    f"token_planes={token_planes}"
                )
            used = seq_planes + rel_planes
            if used > n_planes:
                raise ValueError(
                    f"{DEPTH_MODE} allocation seq={seq_planes} + {rel_planes} depth = {used} "
                    f"exceeds head_dim/2 = {n_planes}"
                )
            # The per-level planes are NOT allocated. That is the arm: everything the tree
            # would have said is meant to be recoverable from depth plus emission order.
            self.depth_used = used
        if self.is_ref_path:
            if ref_planes <= 0:
                raise ValueError(
                    f"{REF_PATH_MODE} needs rope_ref_planes > 0 -- that is the whole arm. Take "
                    "them from the token-leaf planes, which are worthless when sequence rotates."
                )
            if ref_planes % REF_PATH_AXES:
                raise ValueError(
                    f"{REF_PATH_MODE} splits its planes evenly between the ROW and COLUMN axes, "
                    f"so rope_ref_planes must be a multiple of {REF_PATH_AXES}, got {ref_planes}"
                )
            if token_planes:
                raise ValueError(
                    f"{REF_PATH_MODE} isolates the grid-pointer axis so that rope_both is the "
                    f"paired baseline, so it takes no token-leaf planes; got "
                    f"token_planes={token_planes}"
                )
            if levels <= REF_COL_LEVEL:
                raise ValueError(
                    f"{REF_PATH_MODE} reads the enclosing cell's row and column from path "
                    f"levels {REF_ROW_LEVEL} and {REF_COL_LEVEL}, so it needs rope_levels > "
                    f"{REF_COL_LEVEL}, got {levels}"
                )
            used = seq_planes + levels * planes_per_level + ref_planes
            if used > n_planes:
                raise ValueError(
                    f"{REF_PATH_MODE} allocation seq={seq_planes} + {levels} levels x "
                    f"{planes_per_level} + {ref_planes} grid-pointer = {used} exceeds "
                    f"head_dim/2 = {n_planes}"
                )
            self.ref_path_used = used
        if not (self.is_split or self.is_plan or self.is_ref or self.is_rel or self.is_ref_path
                or self.is_depth):
            total = seq_planes + levels * planes_per_level + token_planes
            if total != n_planes:
                raise ValueError(
                    f"plane allocation seq={seq_planes} + {levels} levels x {planes_per_level} + "
                    f"{token_planes} token planes = {total} must equal head_dim/2 = {n_planes}; "
                    "a partial allocation is refused rather than left on sequence position"
                )
            if mode == "rope_leaf" and token_planes == 0:
                raise ValueError("rope_leaf needs token_planes > 0")
        self.mode = mode
        self.slot_planes = slot_planes
        self.rel_planes = rel_planes
        self.ref_planes = ref_planes
        self.n_head = n_head
        self.n_planes = n_planes
        self.seq_planes, self.levels, self.ppl = seq_planes, levels, planes_per_level
        self.token_planes = token_planes
        if self.is_plan:
            # A plan builds only the factors it names, so 'ssssss' costs no tree
            # parameters and 'tttttt' needs no sequence spectrum.
            self.use_seq = "s" in self.plan
            self.use_tree = "t" in self.plan
            self.use_token = self.use_tree          # tree at full width always includes them
        else:
            # rope_ref_anchor rotates EXACTLY what rope_both rotates -- sequence and the tree
            # levels, token planes idle. Its only difference is WHICH coordinate the tree
            # levels carry for a token holding a reference, so it deliberately shares
            # rope_both's allocation check below and adds no group of its own.
            self.use_seq = self.is_split or self.is_ref or self.is_rel or self.is_ref_path or self.anchors_path or self.is_depth or mode in ("rope_seq", "rope_both", "rope_all")
            self.use_tree = self.is_split or self.is_ref or self.is_rel or self.is_ref_path or self.anchors_path or mode in ("rope_graph", "rope_both", "rope_leaf", "rope_all")
            self.use_token = self.is_split or mode in ("rope_leaf", "rope_all")
        self.use_slot = self.is_ref
        # rope_depth shares step_rel and _rel_angles with rope_rel rather than growing a
        # second depth path. use_tree stays False for it, which is the whole difference.
        self.use_rel = self.is_rel or self.is_depth
        self.use_ref_path = self.is_ref_path
        self.use_ref_anchor = self.anchors_path
        self.use_ref_delta = self.is_ref_delta
        inv = base ** (-torch.arange(seq_planes, dtype=torch.float32) / seq_planes)
        self.register_buffer("inv_freq", inv, persistent=False)
        if self.use_tree:
            ladder = [BASE_STEP / (LADDER_RATIO**k) for k in range(planes_per_level)]
            self.step = nn.Parameter(torch.tensor([ladder] * levels, dtype=torch.float32))
            self.theta = nn.Parameter(torch.zeros(levels, planes_per_level))
        if self.use_token:
            tok_ladder = [BASE_STEP / (TOKEN_LADDER_RATIO**p) for p in range(token_planes)]
            self.step_tok = nn.Parameter(torch.tensor(tok_ladder, dtype=torch.float32))
            self.theta_tok = nn.Parameter(torch.zeros(token_planes))
        if self.use_slot:
            slot_ladder = [BASE_STEP / (SLOT_LADDER_RATIO**p) for p in range(slot_planes)]
            self.step_slot = nn.Parameter(torch.tensor(slot_ladder, dtype=torch.float32))
            self.theta_slot = nn.Parameter(torch.zeros(slot_planes))
        if self.use_rel:
            # No theta: every token has a depth, so there is no absent class to separate.
            # Adding one would also break the defining property, because a constant offset
            # per token would survive the parent/child difference only if it were equal for
            # both -- which a presence term is, but a per-DEPTH term would not be.
            rel_ladder = [BASE_STEP / (REL_LADDER_RATIO**p) for p in range(rel_planes)]
            self.step_rel = nn.Parameter(torch.tensor(rel_ladder, dtype=torch.float32))
        if self.use_ref_path:
            # One ladder per axis, shape (2, planes_per_axis): row is index 0, column 1. Both
            # axes start from the same ladder because neither is privileged a priori; training
            # separates them. theta starts at EXACTLY zero, as every other group's does.
            per_axis = ref_planes // REF_PATH_AXES
            ladder = [BASE_STEP / (REF_PATH_LADDER_RATIO**p) for p in range(per_axis)]
            self.step_ref = nn.Parameter(
                torch.tensor([ladder] * REF_PATH_AXES, dtype=torch.float32)
            )
            self.theta_ref = nn.Parameter(torch.zeros(REF_PATH_AXES, per_axis))
        if self.use_ref_delta:
            # EXACTLY zero, so this arm is bit-identical to rope_ref_anchor at step 0 and can
            # only differentiate from it. One constant per (anchor axis, plane within level),
            # matching the shape of the step/theta tables it is added to. The whole result is
            # whether training moves it: nonzero means the model wanted "referent" to be a
            # different state from "co-located", which is the argument this arm exists to test.
            self.delta_ref = nn.Parameter(torch.zeros(REF_ANCHOR_AXES, planes_per_level))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _seq_full(self, t: int, b: int, device) -> torch.Tensor:
        """``(b, t, n_planes)`` -- the sequence spectrum across EVERY plane (split arms)."""
        pos = torch.arange(t, device=device, dtype=torch.float32)
        return (pos[:, None] * self.inv_freq.to(device))[None].expand(b, t, self.n_planes)

    def _tree_full(self, t: int, path, b: int, device) -> torch.Tensor:
        """``(b, t, n_planes)`` -- level planes then token planes, remainder at angle 0.

        Identical arithmetic to ``rope_leaf``; only the layout differs, because here the
        tree owns the head from plane 0 instead of sitting above a sequence band.
        """
        if path is None:
            raise ValueError(f"pos_mode {self.mode!r} needs a DOM path (pos['path'] is None)")
        if path.shape[-1] != self.levels + 1:
            raise ValueError(
                f"pos_mode {self.mode!r} needs the within-node token column "
                f"(path has {path.shape[-1]} columns, expected {self.levels + 1})"
            )
        tree_path = path[..., : self.levels]
        present = (tree_path >= 0).to(torch.float32)[..., None]  # (b,t,L,1)
        rank = tree_path.clamp(min=0).to(torch.float32)[..., None]  # (b,t,L,1)
        ang = rank * self.step[None, None] + present * self.theta[None, None]  # (b,t,L,ppl)
        tree = ang.reshape(b, t, self.levels * self.ppl)
        within = path[..., self.levels]
        present_t = (within >= 0).to(torch.float32)[..., None]  # (b,t,1)
        tok = within.clamp(min=0).to(torch.float32)[..., None] * self.step_tok + present_t * self.theta_tok
        parts = [tree, tok]
        pad = self.n_planes - self.tree_used
        if pad:
            parts.append(torch.zeros(b, t, pad, device=device, dtype=torch.float32))
        return torch.cat(parts, dim=-1)

    def _head_split_angles(self, t: int, path, device) -> torch.Tensor:
        """``(b, n_head, t, n_planes)``: first half of the heads sequence, second half tree."""
        b = path.shape[0] if path is not None else 1
        half = self.n_head // 2
        seq = self._seq_full(t, b, device)[:, None].expand(b, half, t, self.n_planes)
        tree = self._tree_full(t, path, b, device)[:, None].expand(
            b, self.n_head - half, t, self.n_planes
        )
        return torch.cat([seq, tree], dim=1)

    def layer_angles(self, t: int, path, device):
        """``(seq_angles, tree_angles)``, each ``(b, t, n_planes)`` -- for ``rope_layer_split``.

        ``model.py`` hands the first to EVEN layers and the second to ODD layers.
        """
        if self.mode != "rope_layer_split":
            raise ValueError(f"layer_angles is only for rope_layer_split, not {self.mode!r}")
        b = path.shape[0] if path is not None else 1
        return self._seq_full(t, b, device), self._tree_full(t, path, b, device)

    def plan_angles(self, t: int, path, device):
        """``(seq_angles, tree_angles)`` for ``rope_layer_plan``; either may be ``None``.

        ``None`` means the plan never asks for that factor, so it was never built.
        ``model.py`` maps each layer's plan character onto one of these, or onto no
        rotation at all for ``n``.
        """
        if not self.is_plan:
            raise ValueError(f"plan_angles is only for {PLAN_MODE}, not {self.mode!r}")
        b = path.shape[0] if path is not None else 1
        aseq = self._seq_full(t, b, device) if self.use_seq else None
        atree = self._tree_full(t, path, b, device) if self.use_tree else None
        return aseq, atree

    def _slot_angles(self, slot, b: int, t: int, device) -> torch.Tensor:
        """``(b, t, slot_planes)`` -- ``present * theta_slot + slot * step_slot``.

        ``slot`` is ``(B, t)`` int64, ``ABSENT`` where the token carries no symbol or where
        the reader cannot yet know which symbol it is (the gate in ``build_tok_slot``). An
        absent slot contributes EXACTLY zero, as an absent tree level does.
        """
        if slot is None:
            raise ValueError(
                f"pos_mode {self.mode!r} needs the symbol slot (pos['slot'] is None). It comes "
                "from tok_slot.npy, which only a `convert_treepos.py --slots` conversion emits."
            )
        if slot.dim() != 2:
            raise ValueError(f"slot must be (B, t), got shape {tuple(slot.shape)}")
        present = (slot >= 0).to(torch.float32)[..., None]                 # (b,t,1)
        idx = slot.clamp(min=0).to(torch.float32)[..., None]               # (b,t,1)
        return idx * self.step_slot + present * self.theta_slot

    def _rel_angles(self, path, b: int, t: int, device) -> torch.Tensor:
        """``(b, t, rel_planes)`` -- ``depth * step_rel``, the DIRECTED-RELATION group.

        ``depth`` is the number of non-``ABSENT`` levels in the token's root path, so a
        child's depth is its parent's plus one and the angle difference across a parent/child
        hop is EXACTLY ``step_rel`` -- signed, so child-to-parent is ``-step_rel``, and a
        grandparent is ``2 * step_rel``. That is the property the arm exists to test.

        There is deliberately **no presence term**. A token outside the tree has every level
        ``ABSENT`` and therefore depth 0, which is distinct from a depth-1 child of the root;
        it is a real value on the axis rather than a sentinel, and 0.13% of web tokens sit
        there (the ``tokens_in_tree_frac`` in the pack manifest).
        """
        if path is None:
            raise ValueError(
                f"pos_mode {self.mode!r} needs a DOM path (pos['path'] is None); the relational "
                "depth is the count of non-ABSENT levels in it"
            )
        depth = (path[..., : self.levels] >= 0).sum(dim=-1).to(torch.float32)[..., None]
        return depth * self.step_rel

    def _anchor_path(self, tree_path, ref) -> torch.Tensor:
        """``tree_path`` with a reference token's OWN sheet/row/column replaced by its target's.

        This is the whole of ``rope_ref_anchor``. There is no separate group and no new
        parameter: the substituted levels then run through the SAME per-level ``step``/``theta``
        as every other token's, so a reference lands on bit-exactly the angle of the cell it
        names -- the coincidence -- and it lands there on the SAME planes, by the SAME amount,
        that two tokens in one cell agree on. That symmetry is the fix. ``rope_ref_path`` gave
        references a private group, which left same-cell pairs agreeing on 8 planes while a
        pointer and its target agreed on 4 and disagreed on the tree levels.

        ``ref`` is ``(B, t, 3)`` int64 of the target's ``(sheet, row, column)``, ``ABSENT``
        where the token carries no usable reference -- a target that is not a cell, a defined
        name, or a coordinate the builder refused as out of range.

        **A pointer applies only when EVERY axis is present.** A partial overwrite would leave a
        level claiming absence above levels holding real values, which is a malformed path: the
        path is read as a prefix, so anything consuming it mis-reads the token. Requiring all
        three also means there is never a question of what half a coordinate means, because it
        cannot arise.

        This replaces keying on the ROW alone, which ``rope_ref_path`` did to protect whole-row
        targets. Measured over all 52,801,353 token positions of the shipped pack, that guard
        protected a population that does not exist and discarded the one that does:

            row present, column ABSENT :      0     <- what row-keying protected
            column present, row ABSENT :     16     <- what row-keying silently discarded
            both present               : 8,997,782

        The builder now refuses a whole-row target outright and counts it, so the 16 are gone at
        source as well.

        Levels below the column are untouched: they are the token's position *within* the
        formula, which a reference still has.
        """
        if ref is None:
            raise ValueError(
                f"pos_mode {self.mode!r} needs the grid pointer (pos['ref'] is None). It comes "
                "from tok_ref_sheet/row/col.npy, which only build_sheet_pack.py emits."
            )
        if ref.dim() != 3 or ref.shape[-1] != REF_ANCHOR_AXES:
            raise ValueError(
                f"{self.mode} needs ref of shape (B, t, {REF_ANCHOR_AXES}) carrying "
                f"(sheet, row, column); got {tuple(ref.shape)}. A 2-wide pointer is the OLD "
                f"{REF_PATH_MODE} shape and has no sheet axis, which is one of the two faults "
                "this arm exists to fix -- 17.25% of references cross sheets."
            )
        if tree_path.shape[-1] <= REF_COL_LEVEL:
            raise ValueError(
                f"pos_mode {self.mode!r} overwrites path levels {REF_ANCHOR_LEVELS}, but the "
                f"path has only {tree_path.shape[-1]} columns"
            )
        is_ref = self._ref_selector(ref)                                        # (b,t,1)
        head = torch.where(is_ref, ref, tree_path[..., :REF_ANCHOR_AXES])       # (b,t,3)
        return torch.cat([head, tree_path[..., REF_ANCHOR_AXES:]], dim=-1)

    @staticmethod
    def _ref_selector(ref) -> torch.Tensor:
        """``(b, t, 1)`` True where the token carries a COMPLETE pointer -- every axis present.

        ALL axes, not just the row: a partial overwrite is a malformed path, and keying on the
        row alone protected 0 token positions and discarded 16 over the whole shipped pack.
        ``keepdim`` so it broadcasts over the axis dimension.

        Factored out because ``rope_ref_delta`` needs the SAME selector to decide which tokens
        receive the constant. Two copies of this predicate would be free to drift, and a drift
        would mean the overwrite and the offset applied to different token sets -- which would
        look like a working arm.
        """
        return (ref >= 0).all(dim=-1, keepdim=True)

    def _add_ref_delta(self, ang, ref) -> torch.Tensor:
        """Add the learned constant to a REFERENCING token's anchor-level angles only.

        ``ang`` is ``(b, t, levels, ppl)`` straight out of the ordinary per-level arithmetic.
        The constant lands on the first ``REF_ANCHOR_AXES`` levels -- the ones the overwrite
        touched -- and on nothing else, so the token's position within its formula is untouched
        exactly as under the anchor arm.

        ONE-SIDED BY CONSTRUCTION, and that is the property to hold onto. The target is not a
        referencing token, so it does not receive the constant, so
        ``angle(reference) - angle(target) == delta_ref`` exactly. Adding it on both sides would
        cancel in every query-key difference and the arm would silently reduce to
        ``rope_ref_anchor`` -- training, converging, and meaning nothing. That is the same shape
        of failure as giving references a private plane group.
        """
        is_ref = self._ref_selector(ref)[..., None].to(ang.dtype)          # (b,t,1,1)
        pad = self.levels - REF_ANCHOR_AXES
        delta = self.delta_ref
        if pad:
            delta = torch.cat(
                [delta, torch.zeros(pad, self.ppl, device=ang.device, dtype=ang.dtype)], dim=0
            )
        return ang + is_ref * delta[None, None]

    def _ref_path_angles(self, path, ref, b: int, t: int, device) -> torch.Tensor:
        """``(b, t, ref_planes)`` -- the GRID POINTER, written by BOTH sides of a reference.

        ``ref`` is ``(B, t, 2)`` int64 of the TARGET's ``(row, column)``, ``ABSENT`` where the
        token carries no reference. Every token that carries one takes the target's coordinate;
        every other token takes ITS OWN enclosing cell's, out of the path at
        ``REF_ROW_LEVEL`` / ``REF_COL_LEVEL``.

        That single group, written from both sides through the SAME ``step_ref`` /
        ``theta_ref``, is the entire mechanism. Rotary position compares only matching plane
        indices, so giving references a private group would leave the query rotating planes the
        target never writes -- encoding nothing. Here a reference's address tokens land on
        bit-exactly the angle of the cell they name.

        The selector is keyed on the ROW, deliberately: a whole-row target has
        ``ref_col = ABSENT``, and keying per-axis would hand such a token the target's row
        beside its OWN column, which is a coordinate pair that names no cell at all.
        """
        if path is None:
            raise ValueError(
                f"pos_mode {self.mode!r} needs a DOM path (pos['path'] is None); a token's own "
                "cell coordinate is read out of it"
            )
        if ref is None:
            raise ValueError(
                f"pos_mode {self.mode!r} needs the grid pointer (pos['ref'] is None). It comes "
                "from tok_ref_row.npy / tok_ref_col.npy, which only build_sheet_pack.py emits."
            )
        if ref.dim() != 3 or ref.shape[-1] not in (REF_PATH_AXES, REF_ANCHOR_AXES):
            raise ValueError(
                f"ref must be (B, t, {REF_PATH_AXES}) of (row, col) -- or "
                f"(B, t, {REF_ANCHOR_AXES}) of (sheet, row, col) from a pack built after "
                f"2026-09-28, whose trailing two axes are used -- got {tuple(ref.shape)}"
            )
        if ref.shape[-1] == REF_ANCHOR_AXES:
            # A pack carrying the sheet axis. This arm predates it and has no plane for it, so
            # take the (row, col) it does understand. Documented rather than silently sliced:
            # ignoring the sheet is precisely the fault rope_ref_anchor was built to fix.
            ref = ref[..., REF_ANCHOR_AXES - REF_PATH_AXES:]
        if path.shape[-1] <= REF_COL_LEVEL:
            raise ValueError(
                f"pos_mode {self.mode!r} reads path levels {REF_ROW_LEVEL} and {REF_COL_LEVEL}, "
                f"but the path has only {path.shape[-1]} columns"
            )
        own = path[..., REF_ROW_LEVEL : REF_COL_LEVEL + 1]           # (b,t,2)
        is_ref = ref[..., :1] >= 0                                    # (b,t,1), keyed on the row
        coord = torch.where(is_ref, ref, own)                         # (b,t,2)
        present = (coord >= 0).to(torch.float32)[..., None]           # (b,t,2,1)
        idx = coord.clamp(min=0).to(torch.float32)[..., None]         # (b,t,2,1)
        ang = idx * self.step_ref[None, None] + present * self.theta_ref[None, None]
        return ang.reshape(b, t, self.ref_planes)

    def angles(self, t: int, path, device, slot=None, ref=None) -> torch.Tensor:
        """``(B, t, n_planes)`` float32 angles; ``B = 1`` when no path is given.

        ``rope_head_split`` returns ``(B, n_head, t, n_planes)`` instead, because its
        angles differ per head. ``rope_layer_split`` has two angle sets and is refused
        here -- call ``layer_angles`` (or ``forward``) for that arm.

        ``path`` is ``(B, t, levels)`` int64 of child ranks, ``ABSENT`` where the
        node is not that deep or the token is outside the tree -- or
        ``(B, t, levels + 1)`` with the within-node token index as the last
        column, which only ``rope_leaf`` reads.

        ``slot`` is read only by ``rope_ref_slot``; ``ref`` by ``rope_ref_path``,
        ``rope_ref_anchor`` and ``rope_ref_delta``. Every other arm is handed ``None`` for both
        and ignores them, which is what keeps each new pointer arm additive for the arms
        already measured.
        """
        if self.mode == "rope_head_split":
            return self._head_split_angles(t, path, device)
        if self.mode == "rope_layer_split":
            raise ValueError(
                "rope_layer_split has two angle sets; call layer_angles() or forward()"
            )
        if self.is_plan:
            raise ValueError(
                f"{PLAN_MODE} has a per-layer angle set; call plan_angles() or forward()"
            )
        b = path.shape[0] if path is not None else 1
        if self.use_seq:
            pos = torch.arange(t, device=device, dtype=torch.float32)
            seq = (pos[:, None] * self.inv_freq.to(device))[None].expand(b, t, self.seq_planes)
        else:
            seq = torch.zeros(b, t, self.seq_planes, device=device)
        if self.use_tree:
            if path is None:
                raise ValueError(f"pos_mode {self.mode!r} needs a DOM path (pos['path'] is None)")
            if path.shape[-1] not in (self.levels, self.levels + 1):
                raise ValueError(f"path has {path.shape[-1]} columns, module has {self.levels} levels (+1 token column)")
            tree_path = path[..., : self.levels]
            if self.use_ref_anchor:
                # The ENTIRE arm: a reference token's own sheet/row/column become its
                # target's, then the ordinary per-level arithmetic runs unchanged.
                tree_path = self._anchor_path(tree_path, ref)
            present = (tree_path >= 0).to(torch.float32)[..., None]  # (B,t,L,1)
            rank = tree_path.clamp(min=0).to(torch.float32)[..., None]  # (B,t,L,1)
            ang = rank * self.step[None, None] + present * self.theta[None, None]  # (B,t,L,ppl)
            if self.use_ref_delta:
                # R11b: the referencing token ONLY, so reference - target == delta_ref exactly.
                ang = self._add_ref_delta(ang, ref)
            tree = ang.reshape(b, t, self.levels * self.ppl)
        else:
            tree = torch.zeros(b, t, self.levels * self.ppl, device=device)
        if self.use_token:
            if path.shape[-1] != self.levels + 1:
                raise ValueError(f"pos_mode 'rope_leaf' needs the within-node token column (path has {path.shape[-1]} columns)")
            within = path[..., self.levels]
            present_t = (within >= 0).to(torch.float32)[..., None]  # (B,t,1)
            tok = within.clamp(min=0).to(torch.float32)[..., None] * self.step_tok + present_t * self.theta_tok
        else:
            tok = torch.zeros(b, t, self.token_planes, device=device)
        if self.is_depth:
            # rope_depth: sequence, then the depth group, then the remainder UNROTATED. The
            # per-level planes are not emitted at all -- not even as zeros -- so the plane
            # accounting we quote (18 of 64 rotated) is literal and the depth planes sit
            # immediately above sequence rather than above 44 idle ones.
            if path is None:
                raise ValueError(
                    f"pos_mode {self.mode!r} needs a DOM path (pos['path'] is None); the depth "
                    "is the count of non-ABSENT levels in it"
                )
            if path.shape[-1] not in (self.levels, self.levels + 1):
                raise ValueError(
                    f"path has {path.shape[-1]} columns, module has {self.levels} levels "
                    "(+1 token column)"
                )
            parts = [seq, self._rel_angles(path, b, t, device)]
            pad = self.n_planes - self.depth_used
            if pad:
                parts.append(torch.zeros(b, t, pad, device=device, dtype=torch.float32))
            return torch.cat(parts, dim=-1)
        if self.use_rel:
            # rope_rel only: the relational (depth) group, then the remainder unrotated.
            parts = [seq, tree, tok, self._rel_angles(path, b, t, device)]
            pad = self.n_planes - self.rel_used
            if pad:
                parts.append(torch.zeros(b, t, pad, device=device, dtype=torch.float32))
            return torch.cat(parts, dim=-1)
        if self.use_ref_path:
            # rope_ref_path only: the grid-pointer group, then the remainder unrotated.
            parts = [seq, tree, tok, self._ref_path_angles(path, ref, b, t, device)]
            pad = self.n_planes - self.ref_path_used
            if pad:
                parts.append(torch.zeros(b, t, pad, device=device, dtype=torch.float32))
            return torch.cat(parts, dim=-1)
        if not self.use_slot:
            return torch.cat([seq, tree, tok], dim=-1)
        # rope_ref_slot only: the slot group, then any remaining planes left unrotated.
        parts = [seq, tree, tok, self._slot_angles(slot, b, t, device)]
        pad = self.n_planes - self.ref_used
        if pad:
            parts.append(torch.zeros(b, t, pad, device=device, dtype=torch.float32))
        return torch.cat(parts, dim=-1)

    def forward(self, t: int, path, device, slot=None, ref=None):
        if self.is_plan:
            aseq, atree = self.plan_angles(t, path, device)
            pseq = (aseq.cos(), aseq.sin()) if aseq is not None else None
            ptree = (atree.cos(), atree.sin()) if atree is not None else None
            return pseq, ptree
        if self.mode == "rope_layer_split":
            aseq, atree = self.layer_angles(t, path, device)
            return (aseq.cos(), aseq.sin()), (atree.cos(), atree.sin())
        a = self.angles(t, path, device, slot, ref)
        return a.cos(), a.sin()


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate-half RoPE. ``x`` (B, nh, T, hd); ``cos``/``sin`` (B, T, hd/2), shared by
    every head -- or (B, nh, T, hd/2) to give each head its own angles, which is what
    ``rope_head_split`` needs.

    Plane ``i`` pairs dims ``(i, i + hd/2)``.

    PRECISION (2026-09-24). This used to promote ``x`` to float32, rotate, and cast
    back. That was defensive and unnecessary: the ANGLES are still computed in
    float32 (see ``angles``), and ``cos``/``sin`` are bounded in [-1, 1], so the
    rotation is a well-conditioned mix of two values already at the activation's own
    precision -- it cannot lose more than the surrounding matmuls do. Promoting cost
    2.5x here (0.58 ms vs 0.23 ms at B4/H4/T2048/D128), and this runs twice per layer
    per forward, so ~11.5 ms vs ~4.6 ms for a 10-layer model. ``test_rope_pos.py``
    pins the result against the float32 path within bf16 tolerance.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    if cos.dim() == 4:
        # PER-HEAD angles (rope_head_split): already (B, nh, T, hd/2), nothing to broadcast.
        # Added 2026-09-26; the 3-D path below is untouched and stays the hot path for
        # every other arm. `dim()` is a static Python int, so torch.compile traces it away.
        c, s = cos.to(x.dtype), sin.to(x.dtype)
    else:
        c, s = cos[:, None].to(x.dtype), sin[:, None].to(x.dtype)  # (B,1,T,hd/2)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


def read_ref_delta(ckpt_path: str) -> dict:
    """Read ``delta_ref`` out of a ``rope_ref_delta`` checkpoint. THE diagnostic for R11b.

        python -c "import rope_pos,json; print(json.dumps(
            rope_pos.read_ref_delta('$TOPOS_RUNS/sheets/ref_delta_s0/ckpt.pt'), indent=1))"

    ``delta_ref`` initialises to EXACTLY zero, which makes the arm bit-identical to
    ``rope_ref_anchor`` at step 0. So whether it moved is the entire result, and it is read the
    same way the two earlier arms were diagnosed:

      * ``theta_slot`` went 0 -> 1.23 rad on ``rope_ref_slot``, revealing that the model had
        repurposed an identity axis into a presence flag.
      * ``step_ref`` moved 3-6x on ``rope_ref_path``, proving the coordinate was genuinely used
        and so excluding that failure as the explanation for why the arm lost.

    Reading it here:

      * delta stays ~0  -> the model did not want to separate "referent" from "co-located", and
        plain coincidence was already the right encoding. The offset bought nothing.
      * delta moves     -> it did want them separate, which is direct confirmation that the
        anchor's zero difference was confusable with ordinary co-location.

    Returned ``max_abs`` is the number to look at first, and ``base_step_for_scale`` is beside
    it: the tree ladder's own fast step is ``BASE_STEP`` rad per sibling (~0.316), so a delta of
    that order means a full sibling's worth of separation between a referent and a neighbour.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    hits = {k: v for k, v in sd.items() if k.endswith("delta_ref")}
    if not hits:
        raise KeyError(
            f"no delta_ref in {ckpt_path} -- is this a {REF_DELTA_MODE} run? "
            f"rotary parameters present: {sorted(k for k in sd if 'rope' in k)}"
        )
    key, val = next(iter(hits.items()))
    val = val.detach().float()
    return {
        "ckpt": ckpt_path,
        "iter": ck.get("iter_num"),
        "val_nats_tok": ck.get("val_nats_tok"),
        "param": key,
        "shape": list(val.shape),
        "init": 0.0,
        "learned": [[round(float(x), 6) for x in row] for row in val],
        "max_abs": round(float(val.abs().max()), 6),
        "base_step_for_scale": round(BASE_STEP, 6),
        "moved_off_zero": bool(float(val.abs().max()) > 1e-6),
    }
