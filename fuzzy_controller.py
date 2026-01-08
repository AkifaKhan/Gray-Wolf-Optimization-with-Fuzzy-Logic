import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ---------- 1. Define fuzzy input variables ----------
iteration_ratio = ctrl.Antecedent(np.linspace(0, 1, 100), 'iteration_ratio')
diversity = ctrl.Antecedent(np.linspace(0, 1, 100), 'diversity')

# ---------- 2. Define fuzzy output variable ----------
exploration_weight = ctrl.Consequent(np.linspace(0, 2.5, 100), 'exploration_weight')

# ---------- 3. Define membership functions (GBell) ----------
# gbellmf(x, a, b, c)
# a = width (spread), b = slope (steepness), c = center (mean)

# Iteration ratio
iteration_ratio['low'] = fuzz.gbellmf(iteration_ratio.universe, a=0.25, b=4, c=0.0)
iteration_ratio['medium'] = fuzz.gbellmf(iteration_ratio.universe, a=0.25, b=4, c=0.5)
iteration_ratio['high'] = fuzz.gbellmf(iteration_ratio.universe, a=0.25, b=4, c=1.0)

# Diversity
diversity['low'] = fuzz.gbellmf(diversity.universe, a=0.25, b=4, c=0.0)
diversity['medium'] = fuzz.gbellmf(diversity.universe, a=0.25, b=4, c=0.5)
diversity['high'] = fuzz.gbellmf(diversity.universe, a=0.25, b=4, c=1.0)

# Output (exploration weight)
exploration_weight['weak'] = fuzz.gbellmf(exploration_weight.universe, a=0.4, b=4, c=0.5)
exploration_weight['moderate'] = fuzz.gbellmf(exploration_weight.universe, a=0.4, b=4, c=1.2)
exploration_weight['strong'] = fuzz.gbellmf(exploration_weight.universe, a=0.4, b=4, c=1.8)

# ---------- 4. Define fuzzy rules ----------
rules = [
    ctrl.Rule(iteration_ratio['low'] & diversity['high'], exploration_weight['strong']),
    ctrl.Rule(iteration_ratio['medium'] & diversity['high'], exploration_weight['strong']),
    ctrl.Rule(iteration_ratio['low'] & diversity['medium'], exploration_weight['strong']),
    ctrl.Rule(iteration_ratio['medium'] & diversity['medium'], exploration_weight['moderate']),
    ctrl.Rule(iteration_ratio['high'] & diversity['medium'], exploration_weight['moderate']),
    ctrl.Rule(iteration_ratio['high'] & diversity['low'], exploration_weight['weak']),
    ctrl.Rule(iteration_ratio['medium'] & diversity['low'], exploration_weight['weak'])
]

# ---------- 5. Build fuzzy system ----------
fuzzy_ctrl = ctrl.ControlSystem(rules)
fuzzy_sim = ctrl.ControlSystemSimulation(fuzzy_ctrl)

# ---------- 6. Fuzzy–decay hybrid coefficient ----------
def get_fuzzy_a(iter_ratio, diversity_val):
    iter_ratio = np.clip(iter_ratio, 0.001, 0.999)
    diversity_val = np.clip(diversity_val, 0.001, 0.999)

    sim = ctrl.ControlSystemSimulation(fuzzy_ctrl)
    sim.input['iteration_ratio'] = iter_ratio
    sim.input['diversity'] = diversity_val
    sim.compute()

    fuzzy_a = sim.output.get('exploration_weight', 1.0)
    decay_factor = (2 - 2 * iter_ratio)
    hybrid_a = np.clip(fuzzy_a * decay_factor, 0.2, 2.0)
    return hybrid_a

# ---------- 7. Quick sanity check ----------
if __name__ == "__main__":
    print("Iter | Diversity | a_fuzzy")
    for i in range(0, 11):
        ir = i / 10
        div = np.random.uniform(0, 1)
        print(f"{ir:4.1f} | {div:7.3f} → a = {get_fuzzy_a(ir, div):5.3f}")
