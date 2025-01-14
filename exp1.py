#!/usr/bin/env python3

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from tqdm import trange

# ------------------------------------
#  Parameter Mappings
# ------------------------------------
param_mappings = {
    "auction_type": {"first": 1, "second": 0},
    "init": {"random": 0, "zeros": 1},
    "exploration": {"egreedy": 0, "boltzmann": 1},
    "asynchronous": {0: 0, 1: 1},
    "median_opp_past_bid_index": {False: 0, True: 1},
    "winner_bid_index_state": {False: 0, True: 1}
}

# ------------------------------------
#  Payoff with Reserve Price
# ------------------------------------
def get_rewards(bids, auction_type="first", reserve_price=0.0):
    """
    Returns:
      rewards: shape (n_bidders,)
      winner_global: int (index of winner or -1 if no sale)
      winner_bid: float
    """
    n_bidders = len(bids)
    valuations = np.ones(n_bidders)  # placeholder: all valuations = 1
    rewards = np.zeros(n_bidders)

    valid_indices = np.where(bids >= reserve_price)[0]
    if len(valid_indices) == 0:
        return rewards, -1, 0.0  # no sale

    valid_bids = bids[valid_indices]
    sorted_indices = np.argsort(valid_bids)[::-1]
    highest_idx_local = [sorted_indices[0]]
    highest_bid = valid_bids[sorted_indices[0]]

    # Tie-break among top
    for idx_l in sorted_indices[1:]:
        if np.isclose(valid_bids[idx_l], highest_bid):
            highest_idx_local.append(idx_l)
        else:
            break

    # Resolve tie by random choice among highest_idx_local
    if len(highest_idx_local) > 1:
        winner_local = np.random.choice(highest_idx_local)
    else:
        winner_local = highest_idx_local[0]
    winner_global = valid_indices[winner_local]
    winner_bid = bids[winner_global]

    # Find the second-highest among valid
    if len(valid_indices) == len(highest_idx_local):
        second_highest_bid = highest_bid
    else:
        second_idx_local = None
        for idx_l in sorted_indices:
            if idx_l not in highest_idx_local:
                second_idx_local = idx_l
                break
        if second_idx_local is None:
            second_highest_bid = highest_bid
        else:
            second_highest_bid = valid_bids[second_idx_local]

    if auction_type == "first":
        rewards[winner_global] = valuations[winner_global] - winner_bid
    else:  # second-price
        rewards[winner_global] = valuations[winner_global] - second_highest_bid

    return rewards, winner_global, winner_bid

# ------------------------------------
#  State-Space Helpers
# ------------------------------------
def build_state_space(median_opp_past_bid_index, winner_bid_index_state, n_actions):
    """
    If both flags = False -> 1 state
    If exactly one = True -> n_actions
    If both True -> n_actions * n_actions
    """
    if not median_opp_past_bid_index and not winner_bid_index_state:
        return 1
    elif median_opp_past_bid_index and not winner_bid_index_state:
        return n_actions
    elif not median_opp_past_bid_index and winner_bid_index_state:
        return n_actions
    else:
        return n_actions * n_actions

def state_to_index(median_idx, winner_idx, median_flag, winner_flag, n_actions):
    """
    Convert (median_idx, winner_idx) -> single int in [0..n_states-1].
    """
    if not median_flag and not winner_flag:
        return 0
    elif median_flag and not winner_flag:
        return median_idx
    elif not median_flag and winner_flag:
        return winner_idx
    else:
        return median_idx * n_actions + winner_idx

# ------------------------------------
#  Q-Learning Experiment
# ------------------------------------
def run_experiment(alpha, gamma, episodes, auction_type, init, exploration,
                   asynchronous, n_bidders, median_opp_past_bid_index,
                   winner_bid_index_state, reserve_price, seed=0,
                   store_qtables=False, qtable_folder=None):
    """
    Main Q-learning logic for a single (auction_type, alpha, gamma, etc.) experiment.
    Returns summary_dict (final metrics), winning_bids_list, round_history.
    """
    np.random.seed(seed)
    random.seed(seed)

    n_actions = 11
    action_space = np.linspace(0, 1, n_actions)
    n_states = build_state_space(median_opp_past_bid_index, winner_bid_index_state, n_actions)

    # Q-tables initialization
    if init == "random":
        Q = np.random.rand(n_bidders, n_states, n_actions)
    else:
        Q = np.zeros((n_bidders, n_states, n_actions))

    revenues = []
    winning_bids_list = []
    round_history = []

    no_sale_count = 0
    winners_list = []

    window_size = 1000
    start_eps, end_eps = 1.0, 0.0
    decay_end = int(0.9 * episodes)

    prev_bids = np.zeros(n_bidders)
    prev_winner_bid = 0.0

    def bid_to_state_index(bval):
        return np.argmin(np.abs(action_space - bval))

    def get_median_opp_bid_index(bids_array, winner):
        if n_bidders > 1 and winner != -1 and winner < len(bids_array):
            opp_idx = np.delete(np.arange(len(bids_array)), winner)
            median_opp = np.median(bids_array[opp_idx]) if len(opp_idx) > 0 else 0.0
            return bid_to_state_index(median_opp)
        else:
            return 0

    # Initial state
    s = 0
    if median_opp_past_bid_index:
        s = bid_to_state_index(np.median(prev_bids))
    if winner_bid_index_state:
        w_idx = bid_to_state_index(prev_winner_bid)
        if median_opp_past_bid_index:
            s = s * n_actions + w_idx
        else:
            s = w_idx

    save_interval = 1000  # store Q-table snapshots every 1000 episodes

    for ep in range(episodes):
        # Epsilon decay
        if ep < decay_end:
            eps = start_eps - (ep / decay_end) * (start_eps - end_eps)
        else:
            eps = end_eps

        # Each bidder picks an action
        chosen_actions = []
        for i in range(n_bidders):
            if exploration == "boltzmann":
                qvals = Q[i, s]
                exp_q = np.exp(qvals - np.max(qvals))
                probs = exp_q / np.sum(exp_q)
                a_i = np.random.choice(range(n_actions), p=probs)
            else:  # egreedy
                if np.random.rand() > eps:
                    a_i = np.argmax(Q[i, s])
                else:
                    a_i = np.random.randint(n_actions)
            chosen_actions.append(a_i)

        # Convert chosen actions to bids
        bids = np.array([action_space[a] for a in chosen_actions])

        # Evaluate rewards
        rewards, winner, winner_bid_val = get_rewards(bids, auction_type, reserve_price)

        # Seller revenue
        revenue_t = np.max(bids) if winner != -1 else 0.0
        revenues.append(revenue_t)
        winning_bids_list.append(winner_bid_val)
        if winner == -1:
            no_sale_count += 1
        else:
            winners_list.append(winner)

        # Next state
        if median_opp_past_bid_index:
            next_median_idx = get_median_opp_bid_index(bids, winner)
        else:
            next_median_idx = 0
        if winner_bid_index_state:
            next_winner_idx = bid_to_state_index(winner_bid_val) if winner != -1 else 0
        else:
            next_winner_idx = 0

        s_next = state_to_index(next_median_idx, next_winner_idx,
                                median_opp_past_bid_index, winner_bid_index_state,
                                n_actions)

        # Q-update
        if asynchronous == 1:
            # Asynchronous
            for i in range(n_bidders):
                old_q = Q[i, s, chosen_actions[i]]
                td_target = rewards[i] + gamma * np.max(Q[i, s_next])
                Q[i, s, chosen_actions[i]] = old_q + alpha * (td_target - old_q)
        else:
            # Synchronous
            for i in range(n_bidders):
                cf_rewards = np.zeros(n_actions)
                for a_alt in range(n_actions):
                    cf_bids = bids.copy()
                    cf_bids[i] = action_space[a_alt]
                    alt_r, _, _ = get_rewards(cf_bids, auction_type, reserve_price)
                    cf_rewards[a_alt] = alt_r[i]
                max_next_q = np.max(Q[i, s_next])
                Q[i, s, :] = (1 - alpha)*Q[i, s, :] + alpha*(cf_rewards + gamma*max_next_q)

        # Log
        for i in range(n_bidders):
            round_history.append({
                "episode": ep,
                "bidder_id": i,
                "chosen_action_idx": chosen_actions[i],
                "chosen_bid": bids[i],
                "reward": rewards[i],
                "is_winner": (i == winner),
                "auction_type": auction_type,
                "reserve_price": reserve_price
            })

        # Advance
        s = s_next
        prev_bids = bids
        prev_winner_bid = winner_bid_val if winner != -1 else 0.0

        # Optionally store Q-tables
        if store_qtables and (ep % save_interval == 0):
            if qtable_folder is not None:
                np.save(os.path.join(qtable_folder, f"q_after_{ep}.npy"), Q)

    # Final Q snapshot
    if store_qtables and (qtable_folder is not None):
        np.save(os.path.join(qtable_folder, f"q_after_final.npy"), Q)

    # Summaries
    window_size = 1000
    if len(revenues) >= window_size:
        avg_rev_last_1000 = np.mean(revenues[-window_size:])
    else:
        avg_rev_last_1000 = np.mean(revenues)

    rev_series = pd.Series(revenues)
    roll_avg = rev_series.rolling(window=window_size).mean()
    final_rev = avg_rev_last_1000
    lower_band = 0.95 * final_rev
    upper_band = 1.05 * final_rev
    time_to_converge = episodes
    for t in range(len(revenues) - window_size):
        window_val = roll_avg.iloc[t + window_size - 1]
        if lower_band <= window_val <= upper_band:
            stay_in_band = True
            for j in range(t + window_size, len(revenues) - window_size):
                v_j = roll_avg.iloc[j + window_size - 1]
                if not (lower_band <= v_j <= upper_band):
                    stay_in_band = False
                    break
            if stay_in_band:
                time_to_converge = t + window_size
                break

    regrets = [1.0 - r for r in revenues]
    avg_regret_of_seller = np.mean(regrets)
    no_sale_rate = no_sale_count / episodes
    price_volatility = np.std(winning_bids_list)
    if len(winners_list) == 0:
        winner_entropy = 0.0
    else:
        unique_winners, counts = np.unique(winners_list, return_counts=True)
        p = counts / counts.sum()
        winner_entropy = -np.sum(p * np.log(p + 1e-12))

    summary_dict = {
        "avg_rev_last_1000": avg_rev_last_1000,
        "time_to_converge": time_to_converge,
        "avg_regret_of_seller": avg_regret_of_seller,
        "no_sale_rate": no_sale_rate,
        "price_volatility": price_volatility,
        "winner_entropy": winner_entropy,
    }
    return summary_dict, winning_bids_list, round_history

# ------------------------------------
#  Experiment Orchestrator
# ------------------------------------
def run_full_experiment(
    experiment_id=1,
    K=300,
    alpha_choices=[0.05, 0.1],
    gamma_choices=[0.0, 0.5, 0.9, 0.99],
    reserve_price_choices=[0.0, 0.25, 0.5],
    episodes_choices=[10_000, 10_000],
    param_space=None,
    seed=42
):
    """
    Orchestrates the entire experiment end-to-end, given parameter ranges
    and the number of runs (K).
    """
    if param_space is None:
        param_space = {
            "init": ["random", "zeros"],
            "exploration": ["egreedy", "boltzmann"],
            "asynchronous": [0, 1],
            "n_bidders": [2, 4, 6],
            "median_opp_past_bid_index": [False, True],
            "winner_bid_index_state": [False, True]
        }

    random.seed(seed)
    np.random.seed(seed)

    folder_name = f"exp{experiment_id}"
    os.makedirs(folder_name, exist_ok=True)

    # Store param mappings for reference
    with open(os.path.join(folder_name, "param_mappings.json"), "w") as f:
        json.dump(param_mappings, f, indent=2)

    # Folders for logs, plots, Q-tables
    trial_folder = os.path.join(folder_name, "trials")
    os.makedirs(trial_folder, exist_ok=True)

    plots_folder = os.path.join(folder_name, "plots")
    os.makedirs(plots_folder, exist_ok=True)

    q_values_folder = os.path.join(folder_name, "q_values")
    os.makedirs(q_values_folder, exist_ok=True)

    results = []

    # Outer loop: run K times
    for k in trange(K, desc="Overall Experiments"):
        alpha = random.choice(alpha_choices)
        gamma = random.choice(gamma_choices)
        episodes = random.choice(episodes_choices)
        reserve_price = random.choice(reserve_price_choices)
        init_str = random.choice(param_space["init"])
        exploration_str = random.choice(param_space["exploration"])
        asynchronous_val = random.choice(param_space["asynchronous"])
        n_bidders_val = random.choice(param_space["n_bidders"])
        median_flag = random.choice(param_space["median_opp_past_bid_index"])
        winner_flag = random.choice(param_space["winner_bid_index_state"])

        # Subfolder to store Q-tables for this run
        q_run_folder = os.path.join(q_values_folder, f"trial_{k}")
        os.makedirs(q_run_folder, exist_ok=True)

        # We'll collect data for both auctions
        winning_bids_dict = {}

        for auction_type_str in ["first", "second"]:
            # Run experiment
            summary_out, winning_bids, round_hist = run_experiment(
                alpha=alpha,
                gamma=gamma,
                episodes=episodes,
                auction_type=auction_type_str,
                init=init_str,
                exploration=exploration_str,
                asynchronous=asynchronous_val,
                n_bidders=n_bidders_val,
                median_opp_past_bid_index=median_flag,
                winner_bid_index_state=winner_flag,
                reserve_price=reserve_price,
                seed=k,
                store_qtables=True,
                qtable_folder=q_run_folder
            )

            # Annotate round logs
            for row in round_hist:
                row.update({
                    "run_id": k,
                    "alpha": alpha,
                    "gamma": gamma,
                    "episodes": episodes,
                    "reserve_price": reserve_price,
                    "auction_type_code": param_mappings["auction_type"][auction_type_str],
                    "init_code": param_mappings["init"][init_str],
                    "exploration_code": param_mappings["exploration"][exploration_str],
                    "asynchronous_code": param_mappings["asynchronous"][asynchronous_val],
                    "n_bidders": n_bidders_val,
                    "median_opp_past_bid_index_code":
                        param_mappings["median_opp_past_bid_index"][median_flag],
                    "winner_bid_index_state_code":
                        param_mappings["winner_bid_index_state"][winner_flag]
                })

            # Save round-level history
            df_hist = pd.DataFrame(round_hist)
            hist_filename = f"history_run_{k}_{auction_type_str}.csv"
            df_hist.to_csv(os.path.join(trial_folder, hist_filename), index=False)

            # Build final outcome dictionary
            outcome = dict(summary_out)
            outcome["alpha"] = alpha
            outcome["gamma"] = gamma
            outcome["episodes"] = episodes
            outcome["reserve_price"] = reserve_price
            outcome["auction_type_code"] = param_mappings["auction_type"][auction_type_str]
            outcome["init_code"] = param_mappings["init"][init_str]
            outcome["exploration_code"] = param_mappings["exploration"][exploration_str]
            outcome["asynchronous_code"] = param_mappings["asynchronous"][asynchronous_val]
            outcome["n_bidders"] = n_bidders_val
            outcome["median_opp_past_bid_index_code"] = \
                param_mappings["median_opp_past_bid_index"][median_flag]
            outcome["winner_bid_index_state_code"] = \
                param_mappings["winner_bid_index_state"][winner_flag]
            outcome["run_id"] = k

            # Theoretical bounds
            if auction_type_str == "second":
                outcome["theoretical_lower_bound"] = 1.0
                outcome["theoretical_upper_bound"] = 1.0
            else:  # first-price
                if n_bidders_val == 2:
                    outcome["theoretical_lower_bound"] = 0.8
                    outcome["theoretical_upper_bound"] = 1.0
                else:
                    outcome["theoretical_lower_bound"] = 0.9
                    outcome["theoretical_upper_bound"] = 1.0

            results.append(outcome)
            winning_bids_dict[auction_type_str] = winning_bids

        # Plot rolling-average of winning bids (first vs second)
        fig, ax = plt.subplots(figsize=(8, 4))
        window_size = 1000
        for auc_type in ["first", "second"]:
            wb_series = pd.Series(winning_bids_dict[auc_type])
            roll_avg = wb_series.rolling(window=window_size, min_periods=1).mean()
            label_str = "First-Price" if auc_type == "first" else "Second-Price"
            ax.plot(roll_avg, label=label_str)

        title_line_1 = f"Run {k}"
        title_line_2 = (
            f"alpha={alpha}, gamma={gamma}, episodes={episodes}, reserve={reserve_price}, "
            f"init={init_str}, exploration={exploration_str}, async={asynchronous_val}, "
            f"n_bidders={n_bidders_val}"
        )
        ax.set_title(f"{title_line_1}\n{title_line_2}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Rolling Avg Winning Bid (window=1000)")
        ax.legend()
        fig.tight_layout()

        plot_filename = f"plot_run_{k}.png"
        fig.savefig(os.path.join(plots_folder, plot_filename), bbox_inches='tight')
        plt.close(fig)

    # Summaries across all runs
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(folder_name, "data.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"Data generation complete. Saved summary to '{csv_path}'.")


# ------------------------------------
#  Main
# ------------------------------------
if __name__ == "__main__":
    # Example usage: Just call run_full_experiment with the defaults (or set new params)
    run_full_experiment(
        experiment_id=1,
        K=250,  # number of runs
        alpha_choices=[0.001, 0.005, 0.01, 0.05, 0.1],
        gamma_choices=[0.0, 0.5, 0.9, 0.99],
        reserve_price_choices=[0.0, 0.1, 0.2, 0.3],
        episodes_choices=[100_000],
        param_space={
            "init": ["random", "zeros"],
            "exploration": ["egreedy", "boltzmann"],
            "asynchronous": [0, 1],
            "n_bidders": [2, 4, 6],
            "median_opp_past_bid_index": [False, True],
            "winner_bid_index_state": [False, True]
        },
        seed=42
    )
