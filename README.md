# Decision-systems
Practical Work for the Decision Systems and Preferences class of Msc AI at CentraleSupélec


# Sales Representative Assignment and Relocation Optimization

This repository presents a comprehensive solution to the Sales Representative (SR) assignment and relocation problem. The objective is to minimize the total distance traveled by SRs, ensure workload fairness, and manage the number of office relocations for optimal decision-making. The project builds on mathematical optimization techniques, including linear programming, epsilon-constraint methods, and additive preference modeling to derive efficient solutions for the assignment and relocation of SRs to zones (bricks).

## Problem Overview

The problem involves assigning SRs to a set of zones (bricks) in such a way that minimizes both the distance traveled by the SRs and the disruption caused by office relocations. The solution process is framed as a multi-step optimization procedure, gradually increasing in complexity, including handling multi-objective optimization problems and decision-making support systems.

## Key Steps and Solutions

### Step 1: Mono-Objective Optimization Models

In this initial phase, two mono-objective linear optimization models were developed:

1. **Distance Minimization**: A linear program that minimizes the total travel distance for SRs.
2. **Disruption Minimization**: A linear program aimed at minimizing the disruption caused by the assignment of SRs to zones.

The models were implemented using **GUROBI**, a powerful optimization solver, and tested on an instance with 22 bricks and 4 SRs. The optimization was extended using the **epsilon-constraint scheme** to compute a set of non-dominated solutions for the assignment problem under varying workload constraints. This technique allowed us to explore the trade-offs between the two objectives.

- **Workload Constraints**: We implemented workload constraints with different interval values, including `[0.8, 1.2]`, `[0.85, 1.15]`, and `[0.9, 1.1]`.
- **Scalability**: The solution approach was tested with a larger instance of 10 SRs and 100 zones to evaluate the scalability and robustness of the model.

### Step 2: Model Extensions and Partial Assignments

The next step extended the initial model to handle more complex scenarios:

- **Scalability with Larger Instances**: The model was scaled to handle instances with 100 bricks and 10 SRs, and the corresponding sets of non-dominated solutions were computed.
- **Partial Brick Assignment**: A significant modification was introduced to allow a brick to be assigned to multiple SRs. This extension was particularly useful when there was a need to distribute tasks across multiple representatives in a zone. The results of this new model were compared to the original model, providing valuable insights into the trade-offs between flexibility in assignments and the resulting objectives.

Additionally, we explored the scenario where the demand across all bricks increased uniformly by 25%. This required hiring a new SR and determining the optimal location for their office (center brick). This case showcased the adaptability of the model in dynamically adjusting to changes in demand and workforce.

### Step 3: Relocation of Center Bricks

Building on the previous models, Step 3 introduced the relocation of center bricks (offices for SRs) as a variable in the optimization problem:

- **Bi-Objective Optimization**: We formulated a bi-objective problem:
  - **Objective 1**: Minimize the total distance traveled by SRs.
  - **Objective 2**: Minimize the maximum workload among SRs (MinMax).

This approach was particularly useful in balancing the distance minimization with fairness in workload distribution. The challenge was to handle the potential disruption caused by relocating SR offices. A new definition of disruption was introduced: it was quantified by the number of relocated offices, without considering changes in the assignment of bricks to SRs.

- **Three-Objective Problem**: In the final formulation, the disruption measure (number of relocated offices) was integrated with the distance and workload fairness objectives, resulting in a three-objective optimization problem. The non-dominated solutions for this three-objective problem were computed and analyzed to understand the trade-offs between these three competing objectives.

### Step 4: Decision-Support System with UTA Method

To address the decision-making process among the set of non-dominated solutions, we developed an **additive preference model** using the **UTA (UTilités Additives) method**. This method allows for the incorporation of a decision-maker's preferences into the optimization model. 

- **Piecewise Linear Additive Model**: To simulate the preferences of the decision-maker, a piecewise linear model was randomly generated and used as input to the UTA method. This model captures the trade-offs that the decision-maker is willing to make between the three objectives: total distance, workload fairness, and the number of relocated offices.

- **Decision-Making Support**: By integrating the UTA method, we enabled the selection of the most preferred solution from the set of non-dominated solutions. The decision-making process is fully automated, and it ranks solutions based on the decision-maker's preferences for the objectives.

## Solution Approach

- **Mathematical Formulation**: Linear programming models were formulated to minimize the distance and disruption for the assignment problem. The multi-objective optimization problems were addressed using epsilon-constraint methods and bi-objective formulations.
  
- **Optimization Techniques**: GUROBI solver was employed for solving large instances of the linear programs. The epsilon-constraint method was used to explore the non-dominated solutions, while the UTA method provided a structured approach to rank these solutions based on user preferences.

- **Scalability and Flexibility**: The models were designed to be scalable and adaptable to different problem sizes and configurations. This flexibility allows the solution approach to handle changes in the number of SRs, bricks, and the introduction of new constraints or preferences.

## Visualizations and Results

Throughout the project, we included various visualizations to support the analysis:

- **Decision Space Visualizations**: Graphs depicting the trade-offs between objectives, including distance, workload fairness, and disruption.
- **Preference Modeling**: Visualizations showing how the UTA method ranks solutions based on the piecewise linear preferences.
- **Impact of Modifications**: Charts comparing the performance of the models under different configurations (e.g., partial assignments, increased demand, and center brick relocations).

## Conclusion

This project presents a robust and flexible solution for the SR assignment and relocation problem. Through the use of optimization techniques, including linear programming, multi-objective optimization, and decision support systems, we have developed a solution that can handle real-world scenarios with varying constraints and preferences. The approach provides a scalable method for decision-making in complex allocation problems, making it a valuable tool for logistical planning and optimization.
