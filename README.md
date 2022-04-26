> **We provide some pretrained models to show the performance of the proposed method.**
>
> You can find them in ``data/``.

#### For example, in "data/Directions":

- **Moedels:**
  - ``generator_x_4GRU.pkl``
  -  ``generator_y_4GRU.pkl``
  - ``generator_z_4GRU.pkl``
  -  ``generator_v_4GRU.pkl``

-  ``Directions_1.npy`` and `Directions_2.npy` are the pre-processed test files corresponding to the S5 file in the Human 3.6M dataset.

#### Generating prediction sequences:

Change the file read path in `prediction_model.py` to generate the predicted sequence for the specified action and calculate the MPJPE. The generated `GT_X.npy` is the Ground Truth, and `vis_X.npy` is the generated prediction sequence.

#### Generating visualization results:

 Change the file read path and the save path of the generated GIF image in `vis_modle.py` to generate the visualization results.

#### Notes:

1. We train on action class X and test on class X.
2.  Following existing works ( Learning dynamic relationships for 3d human motion prediction et al.),  we use 17 joints to represent a skeleton.

##### **If you have any further questions you can contact us ( *supx19@mails.jlu.edu.cn ).***

