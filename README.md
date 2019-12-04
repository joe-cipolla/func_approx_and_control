# Function Approximation and Control
Here we utilize the Mountain Car Environment to demonstrate:
- Function approximation in the control setting
- Sarsa algo implementation using tile coding
- Tile coding setting changes and subsequent effects on the agent

&nbsp;

&nbsp;

## Mountain Car Environment
In the Mountain Car Environment, the agent is an underpowered car that obeys normal physics. Namely, it is continually being pulled down to the middle of the valley by gravity and it gains momentum the farther it falls. The goal is for the car to reach the top of the mountain to the right, but it does not have enough power to ascend the hill from a cold start at the bottom of the valley. So the car must rock back and forth to gain enough momentum to reach the goal.
&nbsp;

![Mountain Car Environment](https://www.researchgate.net/profile/Marek_Grzes/publication/45107500/figure/fig10/AS:652195724787716@1532506986446/The-mountain-car-task-Sutton-Barto-1998.png)

At each time step the states of the environment are the agent's:
1. current velocity (float between -0.07 and 0.07)
2. current position (float between -1.2 and 0.5)
&nbsp;

The agent has three possible actions:
1. Accelerate right
2. Accelerate left
3. Coast

&nbsp;

&nbsp;

## Tile Coding
The Tile Coding Function provides good generalization and discrimination, consisting of multiple overlapping tilings, where each tiling is a partitioning of the space into tiles.
![Tile Coding](https://www.researchgate.net/profile/Florin_Leon/publication/265110533/figure/fig2/AS:392030699180047@1470478810724/Tile-coding-example.png)

Tile coding is used for function approximation because of the potentially infinite number of states that can occur in the two continuous series of velocity and position (while not exactly infinite at these bounded settings, still large enough to be too expensive to compute).
