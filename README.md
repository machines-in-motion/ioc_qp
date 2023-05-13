# MPC with Sensor-Based Online Cost Adaptation :

This is the code base used for the online cost learning project. The code contains the Bilevel optimization implementation that can generate the optimal
weights for the kuka reaching task w/o obstacles. It also has the pytorch implementations to train the vision encoder and the QPNet. Finally, it contains code
that is used to deploy the MPC algorithm on the real kuka robot. 

The code used to train the QPNet and generate data can be found inside the **Notebook** directory. The code to train the encoder can be found in the **vision** directory. The **demo** directory contains the implementation used to deploy the algorithm on the robot. The **python** directory contains the bilevel optimization problem code along with the various cost functions implemented in python to generate the ground truth reaching motions on the kuka. 

## Dependencies : 

1. pytorch
2. dynamic_graph_head - https://github.com/machines-in-motion/dynamic_graph_head/
3. robot_properties_kuka - https://github.com/machines-in-motion/robot_properties_kuka/

## Citation :

If this code base is used in your research please cite the following paper 
 ```
 @article{meduri2022mpc,
  title={MPC with Sensor-Based Online Cost Adaptation},
  author={Meduri, Avadesh and Zhu, Huaijiang and Jordana, Armand and Righetti, Ludovic},
  journal={arXiv preprint arXiv:2209.09451},
  year={2022}
}
 ```
 
 ## Maintainer :
 
 1. Avadesh Meduri
 2. Huaijiang Zhu

## Copyrights

Copyright(c) 2023 New York University

## License

BSD 3-Clause License
