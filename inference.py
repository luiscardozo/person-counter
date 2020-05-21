#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        self._core = None
        self._network = None
        self._exec_network = None
        self._input_blob_name = None
        self._output_blob_name = None

    def load_model(self, model, device, cpu_extension=None):
        ### Load the model ###
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        self._core = IECore()

        ### Add any necessary extensions ###
        if cpu_extension and "CPU" in device:
            self._core.add_extension(cpu_extension, device)
        
        self._network = IENetwork(model=model_xml, weights=model_bin)

        ### Check for supported layers ###
        unsupported = self.check_unsupported_layers(self._network, device)
        if len(unsupported) != 0:
            print("There are unsupported layers in the current model.\n"
                  "Try to load a CPU extension to solve this problem.\n"
                  "Layer types: " + str(unsupported) + "\n"
                  "Cannot continue. Exiting.", file=sys.stderr)
            exit(1)

        try:
            self._exec_network = self._core.load_network(self._network, device)
        except Exception as e:
            if "unsupported layer" in str(e):
                # OpenVINO throws a RuntimeException on unsupported layer,
                # not an specific type of exception
                print("Cannot run the model, unsupported layer: ", e, file=sys.stderr)
                print("You can try to pass a CPU Extension with the argument --cpu_extension", file=sys.stderr)
            else:
                print(e, file=sys.stderr)
            exit(1)
        
        self._input_blob = next(iter(self._network.inputs))
        self._output_blob = next(iter(self._network.outputs))
        ### Note: You may need to update the function parameters. ###
        return self._exec_network   ########hay que ver si no es un "leak" de implementación

    def get_input_shape(self):
        return self._network.inputs[self._input_blob].shape

    def exec_net(self, image, request_id=0):  #### #Renombrar a async_inference?
        ### TODO: Start an asynchronous request ###
        self._exec_network.start_async(request_id=request_id, inputs={self._input_blob: image})
        ## ???
        # Implement exec_net() by setting a self.infer_request_handle variable
        # to an instance of self.net_plugin.start_async

        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return

    def wait(self, request_id=0, timeout_ms=-1):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return self._exec_network.requests[request_id].wait(timeout_ms) #

    def get_output(self, request_id=0):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return self._exec_network.requests[request_id].outputs[self._output_blob]

    def check_unsupported_layers(self, net, device):
        """
        Checks for unsupported layers on selected device.
        Returns a set of unsupported layers (empty if there are none)
        """
        supported_layers = self._core.query_network(network=net, device_name=device)

        unsupported = set()
        for layer, obj in net.layers.items():
            if not layer in supported_layers and obj.type != 'Input':
                log.debug(f"Unsupported layer: {layer}, Type: {obj.type}")
                unsupported.add(obj.type)

        return unsupported
