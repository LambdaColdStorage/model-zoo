Getting Started
=====================================

* :ref:`create-app`
* :ref:`run-app`


.. _create-app:

Create a machine learning application
--------------------------------------

Model zoo creates machine learning solutions by composing reusable building blocks: 

* **Modeler**: The model pipeline. It encapsulates the forward pass, the computation of loss, the evaluation metric. For reusability, the forward pass is a placeholder that can be dynamically configured to different network architectures. This allows a modeler to be applicable to a general problem. For example, we have image_classification_modeler, machine translation_modeler, object_detection_modeler ... etc.
* **Inputter**: The data pipeline. It reads data from the disk, shuffles and preprocesses the data, creates batches, and does prefetch. Like the modeler, an inputter is applicable to a general problem. For example, we have image_classification_inputter, machine_translation_inputter, object_detection_inputter ... etc.
* **Application**: The excuter. It orchestrates the excution of an inputter and an modeler, distributes the workload across multiple hardware devices, logs the statistics of the job and saves the trained model to disk. 

The value of having the above building blocks is they can be pre-built and re-used in many tasks -- an image_classification_modeler is meant to work with all image classification tasks. The same applies to the inputter. Application is even more general -- it is applicable to many different problems and only varies by the selection of framework and level of APIs.

Now, we can add the core algorithm and complete a machine learning solution:

* **network**: Creates a particular network architecture, such as AlexNet, VGG19, ResNet32 ... etc. It completes the solution by replacing the modeler's placeholder for forward pass.

:numref:`figure-overview` illustrates the composition of a machine learning application.

.. figure:: images/overview.png
   :name: figure-overview

.. _run-app:

Run the applicaiton
--------------------------

blah, blah



