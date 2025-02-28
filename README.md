java c
Optimisinga DNN 
Machine   Learning at Scale   Coursework 
Introduction 
The aim of the coursework for Machine   Learning at   Scale   is to   take   a   DNN   and optimise   it to   improve performance and/or reduce   runtime. We   provide   a working network   implemented in   PyTorch, and associated   input dataset, and then   you   can   experiment with this on Cirrus to try and   improve   performance. The   coursework   is   marked on the report you write, although you will also   submit   the   source   code you develop   in support of your   report.
The   DNN we are using can   be downloaded from the   Learn   page for the   course,   and the data   is already   on   Cirrus   at: 
/scratch/space1/z04/adrianj/mlatscale_coursework
This   location   has already   been   included   in the source code so you should   not need to   modify the code to   use the data. The source   code   for   the   network   is   in   the   coursework_network_online.tar file on the coursework page on Learn. You should download   this to   Cirrus. 
Model 
For the coursework we are using   is an   implementation of a   vision   transformer   (a   good   paper reference for vision transformers is online   here https://arxiv.org/abs/2010.11929). The   vision   transformer   applies   some   of   the features of language transformers to images and   image generation,   particularly   the ability to   include a form. of self-attention that can   let the   model   keep track   of   dependencies   in the   input for future data generation. This   facilitates the generation of images that   include some patterns based   on   previously   seen   data. 

Figure 1: Example of the type of multi-head attention approach used in transformer architectures
This   is   important for our test case, where we are looking at   trying   to   predict weather data   based on previously seen images   of weather fields   (fields are individual weather measurements such as temperature, pressure, wind speeds,   etc…). This   DNN   works   by   reading   in   actual   weather   field   data   and   training   the network to   produce a new output of the field that   represents the   updated weather   in the near future, comparing against the actual   recorded weather data   for   that future time.
There are some   profiling annotations   in the model code   so   that   if you   profile   it      using the   Nvidia   profiling tools we   have used   in   previous exercises you should   get a   breakdown of where the time   is   being spent   in the   model. The   profiling annotations take the form. of the following code additions:
torch.cuda.nvtx.range_push(f"step   {i}")
These should translate into annotated sections   in   profiles you collect. You   can also add your own   profiling   parts   into the code to track specific   sections   in   a   more   fine grained manner   if that would   be   useful. 
Installing Software Prerequisites 
To   run this   model we need to   install and   upgrade some software   packages. Remember, to do this and ensure the   python will work on   the   compute   nodes where the GPUs requires setting a   PYTHONUSERBASE   variable   before you   install   anything   to   install   the   software   in   a   place   where   it   is   accessible,   i.e. the   /work filesystem.   Do the following:
module load nvidia/cudnn/8.6.0-cuda-11.8
module load python/3.10.8-gpu
module load   libsndfile/1.0.28
export PYTHONPATH=$PYTHONPATH:/work/y07/shared/cirrus-software/pytorch/1.13.1-   gpu/python/3.10.8/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/y07/shared/cirrus-software/pytorch/1.13.1-   gpu/python/3.10.8/lib
export LIBRARY_PATH=$LIBRARY_PATH:/work/y07/shared/cirrus-software/pytorch/1.13.1-   gpu/python/3.10.8/lib
export PYTHONUSERBASE=/work/m24ol/m24ol/$USER/python-installs
python3 -m pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0   --index-url
https://download.pytorch.org/whl/cu118 
python -m pip   install protobuf==3.20
python -m pip   install   --upgrade tensorboard
python3 -m pip   install   h5py
You   can   change   the   location   of   the   install   (i.e. what   is   set   in   PYTHONUSERBASE)   provided   it   is   on   the   /work   filesystem,   however,   if   you   do   this   you   will   need   to update the   batch script. we   have   provided to   point to that   location.
Running the model 
You   have   been   provided with a   batch script. to submit the   model and   run with a reduced input set to enable relatively swift   experimentation   (run_coursework.sh).   In   there   you   will   see   that   we   call   the   DNN   with   a   parameter,   i.e.:
python3 train.py   --config   shortThe   parameter short sets   up the   reduced   input   runs,   but you can also change   it to base   to   run the full   input dataset and   model.   Both these   parameters select the configuration from the file in   config/coursework_transformer.yaml.   You   are   not   necessarily   required to change this configuration,   but you can see   the types of things that are specified   in the yaml file, and also   add   new   parameters   to   that file   if you want to   be able to control and runtime things   you   add   to   the   model      (i.e.   parallelisation   approaches, etc…).   Changing   the   configuration   can   be   a   vali代 写Optimising a DNN Machine Learning at Scale CourseworkPython
代做程序编程语言d approach for optimisation   if you can justify why you’re changing it and what performance benefits you expect the change   to   enable.
The   short   configuration   should   run   around   4 training   epochs   and   complete   in   10   to   20   minutes. You   can   make   this   even   shorter   if   required,   by   reducing   either limit_nsamples   or   num_iters   in   the   input   file. The   number   of   epochs   to   use is automatically calculated   by dividing the number of the   number of   iterations (num_iters)   by   the   data   samples   to   be   processes   per   batch (limit_nsamples/   global_batch_size).   For   the   full   run   (the   base configuration) around   38,000 samples   are   used   with   a   batch   size   of   16,   giving   the data   samples   per   batch   as   around   2375, and   the   current   configuration   uses   a num_iters   30,000,   meaning   the   model   will   run   12 epochs   for   training.
When you   run the code you should get output   like this   in your   batch   script.
2024-03-08 02:30:33,527 -   root   -   INFO   -   Time   taken   for   epoch   2   is
10622.394329 sec, avg   3.572829   samples/sec
2024-03-08 02:30:33,633 -   root   -   INFO   -       Avg   train   loss=0.166270
2024-03-08 02:36:08,512 - root   -   INFO   -       Avg   val   loss=0.15351414680480957
2024-03-08 02:36:08,526 -   root   -   INFO   -         Total   validation   time:
299.874281167984   sec
You can find   more detailed   information   in the files   in the   logs   directory, particularly the   out.log   file for the associated   run you are looking at.   Note,      these log files   may get overwritten every time you   run the   model,   so   you   may   want to copy them to a different location   if you want to   keep   them for future analysis/evaluations.
Profiling 
As   part of the coursework   it   is sensible to   profile the   network to see where the performance   bottlenecks are. You can   use the   nvidia   nsight tools, and dlprof, as   outlined   in   previous exercises on the course.   If you encounter an issue   profiling      where you get error messages such   as   the   following: 
ModuleNotFoundError: No module named   'fast_multihead_attn'.
You can fix this   by   installing the following   profiling add on which   is   included   in   some version of PyTorch,   but   may   not   be available   in those   install on   Cirrus:
cd /work/m24ol/m24ol/$USER
git   clone https://github.com/NVIDIA/apex 
cd   apex/
pip install -v --no-build-isolation   --no-cache-dir   --global-   option="--cpp_ext" --global-option="--cuda_ext" --global-
ption="--fast_multihead_attn"   ./
Assignment 
The coursework tasks are for you to try and speed   up the   model,   either   by making adjustments and   improvements to the code and   how   it   is   run   in   PyTorch,   or by adding   parallelisation to the   model, or by   doing   both   of this. As with   any optimisation approach, you should start with   profiling the model to   see where   time   is   being spent and use that data   to   guide   optimisations you   undertake. 
The coursework   is   marked on the report   you submit. We   are   expecting   a   report   of   around   10   pages   that   outlines   the   profiling   and   initial   performance   of   the   model, and then any optimisation work you   have undertaken and what the   outcomes   of   that optimisations were. There are no required or expected   optimisations,   any   sensible   approach   at   optimising will be acceptable,   provided you document and justify this   in the   report.
You should also consider the impact of any   changes you   make   on   the   overall quality of the   predictions the   DNN   produces. This means you should document the   impact on the training and validation losses that changes   cause,   and   discuss whether those changes should be kept/implemented or   not   given   the   impacts   on   network prediction   quality.
Please ensure that you include your exam number in the title of both your report and your source code. This assignment will be marked anonymously so we cannot identify which report goes with which source code unless you include your exam number in the title. 
Marking scheme 
The   report will   be   marked on:
•         Discussion of the   performance of the   DNN, optimisations   proposed and undertaken,   performance achieved, and   impact on the quality of the   DNN   results   (i.e.   impact   on   loss   metrics) (70).
•         Methodology used   in the assignment as demonstrated   in the   report. This   includes   general   approach, tools   used   etc (20).
•         Clarity, relevance   and   presentation   of   the   report (10).
This   coursework   is   due   at 11.59, Monday 17th March 2025 (UK Time) 
As per the   University's Taught Assessment   Regulations (for further information see   link on   Learn course Assessment page) assignments submitted after   the deadline (unless granted an extension, see Student   Support   page   on   the   Learn course) are   subject   to   a   5%   penalty   per   day   (i.e. 24   hours)   that   the   assignment   is late after the deadline, up to a   maximum of seven. Assignments   handed   in   more      than seven days late   receive   zero   marks. 





         
加QQ：99515681  WX：codinghelp  Email: 99515681@qq.com
