#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <iomanip>

// A Q-Learning Agent class 
class QLearningAgent {
private:
    int num_states;
    int num_actions;
    double alpha;    // Learning rate
    double gamma;    // Discount factor
    double epsilon;  // Exploration rate
    
    // 2D Vector representing Q-Table
    std::vector<std::vector<double>> q_table;
    
    // Random Number Generation
    std::mt19937 rng; 

public:
    // Initialize the Q-table with zeros
    QLearningAgent(int states, int actions, double a, double g, double e)
        : num_states(states), num_actions(actions), alpha(a), gamma(g), epsilon(e) {
        
        // Initialize Q-table: states x actions grid of 0.0s
        q_table.resize(num_states, std::vector<double>(num_actions, 0.0));
        
        // Seed the random number generator
        std::random_device rd;
        rng = std::mt19937(rd());
    }

    // Epsilon-Greedy action selection
    int chooseAction(int state) {
        std::uniform_real_distribution<double> dist_real(0.0, 1.0);
        
        if (dist_real(rng) < epsilon) {
            // Explore: Choose random action
            std::uniform_int_distribution<int> dist_int(0, num_actions - 1);
            return dist_int(rng);
        } else {
            // Exploit: Choose action with highest Q-value
            return (q_table[state][0] > q_table[state][1]) ? 0 : 1;
        }
    }

    // Bellman Equation update
    void updateQValue(int state, int action, double reward, int next_state) {
        double max_next_q = std::max(q_table[next_state][0], q_table[next_state][1]);
        
        q_table[state][action] = q_table[state][action] + 
                                 alpha * (reward + gamma * max_next_q - q_table[state][action]);
    }

    // Utility to visualize the learned policies
    void printQTable() const {
        std::cout << "\nLearned Q-Table (State Values):\n";
        std::cout << "-----------------------------------\n";
        for (int s = 0; s < num_states; ++s) {
            std::cout << "State " << s << " | Left: " << std::fixed << std::setprecision(2) 
                      << q_table[s][0] << " | Right: " << q_table[s][1] << "\n";
        }
    }
};

int main() {
    // 1D Path with 6 states (0 to 5)
    // State 0 is a pit (Penalty), State 5 is the Goal (Reward)
    // Actions: 0 = Move Left, 1 = Move Right
    const int NUM_STATES = 6;
    const int NUM_ACTIONS = 2; 
    const int GOAL_STATE = 5;
    const int PIT_STATE = 0;
    const int EPISODES = 500;

    // Initialize Agent (States, Actions, Alpha, Gamma, Epsilon)
    QLearningAgent agent(NUM_STATES, NUM_ACTIONS, 0.1, 0.9, 0.2);

    std::cout << "Training Agent for " << EPISODES << " episodes...\n";

    // Training Loop
    for (int i = 0; i < EPISODES; ++i) {
        int state = 2; // Start in the middle of the grid
        
        while (state != GOAL_STATE && state != PIT_STATE) {
            int action = agent.chooseAction(state);

            // Simulate the environment's response
            int next_state = state;
            if (action == 0) next_state--; // Move Left
            if (action == 1) next_state++; // Move Right

            // Calculate Reward
            double reward = -0.1; // Small penalty for each step to encourage fast completion
            if (next_state == GOAL_STATE) reward = 10.0;
            if (next_state == PIT_STATE) reward = -10.0; 

            // Agent learns from the transition
            agent.updateQValue(state, action, reward, next_state);

            // Move to next state
            state = next_state;
        }
    }

    // Output results
    agent.printQTable();
    std::cout << "\nNotice how states closer to the Goal (State 5) have higher 'Right' values,\n";
    std::cout << "and states closer to the Pit (State 0) strongly prefer moving 'Right'.\n";

    return 0;
}
