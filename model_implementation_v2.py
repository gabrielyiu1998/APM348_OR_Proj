# Model Implementation Ice Cream Production

# Libraries to Import
import matplotlib.pyplot as plt
import numpy as np 
import scipy.stats as st
import operator
import time

# Parameters

B = 100 # order to be filled
M = 50 # maximum production per day
t_D = 10 # deadline in days
C_I = 4 # Cost of inventory per day per unit of ice cream
C_L = 4 # Cost of being late per day
p_b = 0.02 # Probability of breakdown per day
p_r = 0.7 # Probability of repair per day

# Model Implementation
def C_p(delta_xt):
    """
    Parameters
    ----------
    delta_xt : Int
        Amount the policy suggests.

    Returns
    -------
    Int
        Cost of producing delta_xt ice cream.
    """
    
    return delta_xt ** 2

def incurred_day_cost(inventory, production, t_until_d):
    """
    Parameters
    ----------
    inventory : Int
        Amount in inventory at the beginning of day time.
    production : Int
        Amount produced during day time.
    time : Int
        Time at beginning of day.

    Returns
    -------
    Int
        Total costs incurred during day time.
    """
    
    return inventory * C_I + C_p(production) + (t_until_d == 0) * C_L

# Q Learning Implementation
def qlearning(epsilon, learning_rate, trials = 100000, one_sim = False):
    """
    Creates the q-learning table.

    Parameters
    ----------
    epsilon : float
        0 < epsilon < 1, explore vs exploit.
    learning_rate : float
        Rate at which new value overwrites old value.
    trials: int, optional
        Number of episodes to train on. The default is 100000
        corresponds to a run time of about one minute.
    one_sim: bool, optional
        Option to plot the learning process. The default is False.

    Returns
    -------
    dict
        The policy that describes your action.

    """
    start = time.time()
    Q_table = {}
    for state in range(2):
        for inven in range(B + 1):
            for days in range(t_D + 1):
                if state == 0:
                    Q_table[(state, inven, days)] = {0: 0}
                else:
                    av = {}
                    for action in range(min(B - inven + 1, M + 1)):
                        av[action] = -incurred_day_cost(B, M, 0)
                    Q_table[(state, inven, days)] = av
    
    def run_episode(days_td, st_inv, epsil, lr):
        machine_state_t, inventory, t_until_d = 1, st_inv, days_td
        cost = 0
        while inventory < B or t_until_d > 0:
            if machine_state_t == 1:
                if np.random.uniform() < epsil:
                    d_xt = np.random.randint(0, min([M + 1,
                                                     B - inventory + 1]))
                else:
                    d_xt = max(Q_table[(1, inventory, t_until_d)].items(),
                                        key=operator.itemgetter(1))[0]
                machine_state_t_1 = np.random.binomial(1, 1 - p_b)
                if machine_state_t_1 == 1:
                    inventory_new = int(inventory + d_xt)
                else:
                    inventory_new = inventory + np.random.randint(0, d_xt + 1)
            else:
                d_xt = 0
                machine_state_t_1 = np.random.binomial(1, p_r)
                inventory_new = inventory
            cost_t = incurred_day_cost(inventory, d_xt, t_until_d)
            cost += cost_t
            old_value = Q_table[(machine_state_t, inventory, t_until_d)][d_xt]
            Q_table[(machine_state_t, inventory, t_until_d)][d_xt] += lr * (
                -cost_t +
                max(Q_table[(machine_state_t_1, inventory_new,
                             max([0, t_until_d - 1]))].values())
                - old_value)
            inventory = inventory_new
            machine_state_t = machine_state_t_1
            t_until_d = max([0, t_until_d - 1])
        return cost
    
    costs = []
    for ep in range(trials):
        costs.append(run_episode(t_D, 0, epsilon, learning_rate))
    
    end = time.time()
    mins = np.floor((end - start) / 60)
    sec = (end - start) - mins * 60
    print("Run Time: {} min, {} sec".format(mins, round(sec, 1)))
    
    if one_sim:
        tc = one_simulation(Q_table, True)
        print(tc) 
        plt.plot(np.arange(len(costs)), costs)
        
    return Q_table

# Simulation

def one_simulation(policy, verbose = True, init_inv = 0, time_until_d = t_D):
    """
    Runs one simulation based on the policy.

    Parameters
    ----------
    policy : dict
        Probably the output of qlearning.
    init_inv : int, optional
        Initial inventory. The default is 0.
    time_until_d : int, optional
        Time until deadline. The default is t_D.
    verbose : bool, optional
        Option to print statements on each day of production.
        The default is False.

    Returns
    -------
    int
        Total cost.

    """
    machine_state_t, inventory, t_until_d = 1, init_inv, time_until_d
    time, cost = 0, 0
    prod_per_day = []
    states = []
    while inventory < B or t_until_d > 0:
        states.append(machine_state_t)
        if machine_state_t == 1:
            delta_xt = max(policy[(1, inventory, t_until_d)].items(),
                           key=operator.itemgetter(1))[0]
            if delta_xt == 0 and t_until_d == 0:
                raise RuntimeError("Policy will never finish production.")
            machine_state_t = np.random.binomial(1, 1 - p_b)
            if machine_state_t == 1:
                inventory_new = int(inventory + delta_xt)
            else:
                inventory_new = inventory + np.random.randint(0, delta_xt + 1)
        else:
            delta_xt = 0
            machine_state_t = np.random.binomial(1, p_r)
            inventory_new = inventory
        prod_per_day.append(inventory_new - inventory)
        cost += incurred_day_cost(inventory, delta_xt, t_until_d)
        inventory = inventory_new
        if verbose:
            print("End of day {}.".format(time))
            print("Machine State: {}".format(machine_state_t))
            print("Decision made: {}".format(delta_xt))
            print("Actual Production: {}".format(prod_per_day[time]))
            print("Inventory: {} \n".format(inventory_new))
        t_until_d = max([0, t_until_d - 1])
        time += 1
    return cost, prod_per_day, states

if __name__ == '__main__':
    np.random.seed(348)
    
    robustness = False
    sensitivities = True
    
    
    # Parameters
    B = 100 # order to be filled
    M = 50 # maximum production per day
    t_D = 10 # deadline in days
    C_I = 4 # Cost of inventory per day per unit of ice cream
    C_L = 1000 # Cost of being late per day
    p_b = 0.02 # Probability of breakdown per day
    p_r = 0.7 # Probability of repair per day
        
    # Robustness
    if robustness:    
        # Deterministic Schedule
        print("Deterministic Schedule")
        p_b = 0
        C_L = 50
        Q_table_det = qlearning(0.2, 0.8)
        costd, scheduled, statesd = one_simulation(Q_table_det)
        print("Total Cost: {}".format(costd))
        print("Schedule: {}".format(scheduled))
        print("States: {}".format(statesd))
        
        plt.figure("Robustness Plot")
        plt.step(np.arange(len(scheduled)), np.cumsum(scheduled),
                 label = "$p_B = 0.00$", where = "post", alpha=0.5)
        
        # Small Probability of Breakdown
        print("\n Small probability of breakdown.")
        p_b = 0.01
        Q_table_spb = qlearning(0.2, 0.8)
        cost1, schedule1, states1 = one_simulation(Q_table_spb)
        breakdown = any([item == 0 for item in states1])
        if breakdown:
            bd = "Breakdown Occured"
        else:
            bd = "No Breakdown"
            
        plt.step(np.arange(len(schedule1)), np.cumsum(schedule1),
                 label = "$p_B = 0.01$" + " ("+ bd + ")", where = "post",
                 alpha=0.5)
        print(bd)
        print("Total Cost: {}".format(cost1))
        print("Schedule: {}".format(schedule1))
        print("States: {}".format(states1))
        
        if breakdown:
            while breakdown:
                cost2, schedule2, states2 = one_simulation(Q_table_spb, False)
                breakdown = any([item == 0 for item in states2])
            bd = "No Breakdown"
        else:
            while not breakdown:
                cost2, schedule2, states2 = one_simulation(Q_table_spb, False)
                breakdown = any([item == 0 for item in states2])
            bd = "Breakdown Occured"
        
        print("\n " + bd)
        print("Total Cost: {}".format(cost2))
        print("Schedule: {}".format(schedule2))
        print("States: {}".format(states2))
        
        plt.step(np.arange(len(schedule2)), np.cumsum(schedule2),
                 label = "$p_B = 0.01$" + " ("+ bd + ")", where = "post",
                 alpha=0.5)
        plt.xlabel("t (Days)")
        plt.ylabel("x(t) Buckets")
        plt.title("Robustness")
        plt.legend()      

        
        # Different learning outcomes when p_B > 0
        print("\n Different learning outcomes")
        p_b = 0.05
        C_I = 8
        C_L = 500
        Q_table1 = qlearning(0.2, 0.8, 150000)
        Q_table2 = qlearning(0.2, 0.8, 150000)
        costs1 = [one_simulation(Q_table1, False)[0] for i in range(100)]
        print("Mode of costs for policy one: {}".format(st.mode(costs1)[0][0]))
        print(st.describe(costs1))
        costs2 = [one_simulation(Q_table2, False)[0] for i in range(100)]
        print("Mode of costs for policy two`: {}".format(st.mode(costs2)[0][0]))
        print(st.describe(costs2))
        plt.figure("Two Policies Same Parameters")
        plt.xlabel("Costs")
        plt.ylabel("Frequency")
        c1_bins =  plt.hist(costs1, bins = 10,
                            alpha = 0.5, label = "Policy 1")[1]
        plt.hist(costs2, bins = c1_bins, alpha = 0.5, label = "Policy 2")
        plt.legend()
    
    if sensitivities:
        policies = {}
        for C_L in [1000, 2000, 3000]:
            plt.figure("Sensitivities {}".format(C_L))
            for p_b in [0.01, 0.05, 0.1]:
                policies[(C_L, p_b)] = qlearning(0.2, 0.3, 250000)
                cost_t, sch_t, sta_t =one_simulation(policies[(C_L, p_b)],
                                                     False)
                while any([i == 0 for i in sta_t]):
                    cost_t, sch_t, sta_t =one_simulation(policies[(C_L, p_b)],
                                                         False)
                print("Late Fee {}, P(Breakdown) {}".format(C_L, p_b))
                print("Total Cost: {}".format(cost_t))
                print("Schedule: {}".format(sch_t))
                print("States: {}".format(sta_t))
                plt.plot(np.arange(len(sch_t)), np.cumsum(sch_t),
                         label = "$p_B$ ={}".format(p_b),
                         alpha=0.5)
                print("\n")
            plt.xlabel("t (Days)")
            plt.ylabel("x(t) Buckets")
            plt.title("Schedules $C_L$={}".format(C_L))
            plt.legend()

        

        