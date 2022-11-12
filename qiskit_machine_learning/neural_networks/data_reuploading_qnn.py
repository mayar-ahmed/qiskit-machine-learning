# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Description here"""
from __future__ import annotations

import numpy as np

from qiskit import QuantumCircuit, ParameterVector
from qiskit.opflow import PauliSumOp, StateFn, OperatorBase, ExpectationBase
from qiskit.providers import Backend
from qiskit.utils import QuantumInstance

from .opflow_qnn import OpflowQNN
from ..utils import derive_num_qubits_feature_map_ansatz


class Reuploading1Q(OpflowQNN):
    """Two Layer Quantum Neural Network consisting of a feature map, a ansatz,
    and an observable.
    """

    def __init__(
        self,
        num_layers: int | None = None,
        num_features: int | None = None,
        num_classes:int | None= None,
        class_repr: np.ndarray | None = None,
        observable: OperatorBase | QuantumCircuit | None = None,
        exp_val: ExpectationBase | None = None,
        quantum_instance: QuantumInstance | Backend | None = None,
        input_gradients: bool = False,
    ):
        r"""
        Args:
            num_layers: 
            num_features: 
            num_classes: 
            observable: observable to be measured to determine the output of the network. If
                ``None`` is given, the :math:`Z^{\otimes num\_qubits}` observable is used.
            exp_val: The Expected Value converter to be used for the operator obtained from the
                feature map and ansatz.
            quantum_instance: The quantum instance to evaluate the network.
            input_gradients: Determines whether to compute gradients with respect to input data.
                Note that this parameter is ``False`` by default, and must be explicitly set to
                ``True`` for a proper gradient computation when using
                :class:`~qiskit_machine_learning.connectors.TorchConnector`.
        Raises:
            QiskitMachineLearningError: Needs at least one out of ``num_qubits``, ``feature_map`` or
                ``ansatz`` to be given. Or the number of qubits in the feature map and/or ansatz
                can't be adjusted to ``num_qubits``.
        """

        # generate parameters
        #how are we going to feed this input? 
        input_params= ParameterVector('input', num_features)

        #create vector with trainable parameters
        n_u3= int(np.ceil(num_features/3))
        #to be extended to multi-qubits
        n_qubits =  1
        n_theta=  n_qubits* num_layers*3*n_u3
        n_alpha= n_qubits*num_layers*num_features
        
        weight_params= ParameterVector('weights', n_theta+n_alpha)

        #divide into theta and alpha to create circuit
        theta = np.array(weight_params[:n_theta]).reshape(n_qubits, num_layers, 3*n_u3)
        alpha = np.array(weight_params[n_theta:]).reshape(n_qubits, num_layers, num_features)
        
        theta_alpha  =theta.copy()
        #add theta and alpha for each ugate
        self._circuit = QuantumCircuit(1)
        for n in range(n_qubits):
            for l in range(num_layers):
                    #add theta and phi for the ugates
                    for i in range(num_features):
                        #map this to each u layer
                        theta_alpha[n][l][i] += alpha[n][l][i]*input_params[i]
            

        #add u3 gates for each layer in the circuit
        for l in range(num_layers):
            u_params = theta_alpha[1, l,:]
            for u in range(n_u3):
                self._circuit.u3(u_params[u*3], u_params[u*3+1], u_params[u*3+2])


        # construct observable
        self.observable = (
            observable if observable is not None else PauliSumOp.from_list([("Z" * 1, 1)])
        )

        # combine all to operator
        operator = StateFn(self.observable, is_measurement=True) @ StateFn(self._circuit)

        super().__init__(
            operator=operator,
            input_params=input_params,
            weight_params=weight_params,
            exp_val=exp_val,
            quantum_instance=quantum_instance,
            input_gradients=input_gradients,
        )


    @property
    def circuit(self) -> QuantumCircuit:
        """Returns the underlying quantum circuit."""
        return self._circuit

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits used by ansatz and feature map."""
        return self._circuit.num_qubits
