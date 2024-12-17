import greedy3
from qc import rand_state, print_state

print(greedy3.templates(5))

patterns = greedy3.patterns(5)
for p in patterns:
    print(p, "  ", greedy3.pattern_cx_cost(p))

state = rand_state(4)
tracker = greedy3.StateTracker(state)

print_state(tracker.state)
print()
tracker.rotate_merge(7, 3)
print_state(tracker.state)
print()
tracker.undo_gate()
tracker.undo_gate()
print_state(tracker.state)
print()
tracker.rotate_merge(15, 7)
tracker.rotate_merge(7, 3)
tracker.rotate_merge(3, 1)
tracker.rotate_merge(1, 0)
print_state(tracker.state)
print()
tracker.new_block()
tracker.control_rotate_merge(12, 8)
print_state(tracker.state)
print()
tracker.undo_block()
print_state(tracker.state)
print()
tracker.new_block()
tracker.rotate_merge_chunk("11**", 1)
print_state(tracker.state)
print()
tracker.undo_block()
tracker.control_rotate_merge_chunk("11**", 0, 1)
print_state(tracker.state)

