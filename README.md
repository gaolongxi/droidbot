![DroidBot UTG](droidbot/resources/dummy_documents/droidbot_utg.png)

# DroidBot

## New!

:fire: We recently integrated ChatGPT into DroidBot to support automating any app with a simple text prompt. [Take a look!](https://github.com/GAIR-team/DroidBot-GPT)


## About
DroidBot is a lightweight test input generator for Android.
It can send random or scripted input events to an Android app, achieve higher test coverage more quickly, and generate a UI transition graph (UTG) after testing.

A sample UTG is shown [here](http://honeynet.github.io/droidbot/report_com.yelp.android/).

DroidBot has the following advantages as compared with other input generators:

1. It does not require system modification or app instrumentation;
2. Events are based on a GUI model (instead of random);
3. It is programmable (can customize input for certain UI);
4. It can produce UI structures and method traces for analysis.

**Reference**

[Li, Yuanchun, et al. "DroidBot: a lightweight UI-guided test input generator for Android." In Proceedings of the 39th International Conference on Software Engineering Companion (ICSE-C '17). Buenos Aires, Argentina, 2017.](http://dl.acm.org/citation.cfm?id=3098352)

## Prerequisite

1. `Python` (both 2 and 3 are supported)
2. `Java`
3. `Android SDK`
4. Add `platform_tools` directory in Android SDK to `PATH`
5. (Optional) `OpenCV-Python` if you want to run DroidBot in cv mode.

## How to install

Clone this repo and install with `pip`:

```shell
git clone https://github.com/honeynet/droidbot.git
cd droidbot/
pip install -e .
```

If successfully installed, you should be able to execute `droidbot -h`.

## How to use

1. Make sure you have:

    + `.apk` file path of the app you want to analyze.
    + A device or an emulator connected to your host machine via `adb`.

2. Start DroidBot:

    ```
    droidbot -a <path_to_apk> -o output_dir
    ```
    That's it! You will find much useful information, including the UTG, generated in the output dir.

    + If you are using multiple devices, you may need to use `-d <device_serial>` to specify the target device. The easiest way to determine a device's serial number is calling `adb devices`.
    + On some devices, you may need to manually turn on accessibility service for DroidBot (required by DroidBot to get current view hierarchy).
    + If you want to test a large scale of apps, you may want to add `-keep_env` option to avoid re-installing the test environment every time.
    + You can also use a json-format script to customize input for certain states. Here are some [script samples](script_samples/). Simply use `-script <path_to_script.json>` to use DroidBot with a script.
    + If your apps do not support getting views through Accessibility (e.g., most games based on Cocos2d, Unity3d), you may find `-cv` option helpful.
    + You can use `-humanoid` option to let DroidBot communicate with [Humanoid](https://github.com/yzygitzh/Humanoid) in order to generate human-like test inputs.
    + You may find other useful features in `droidbot -h`.

## Evaluation

We have conducted several experiments to evaluate DroidBot by testing apps with DroidBot and Monkey.
The results can be found at [DroidBot Posts](http://honeynet.github.io/droidbot/).
A sample evaluation report can be found [here](http://honeynet.github.io/droidbot/2015/07/30/Evaluation_Report_2015-07-30_1501.html).

## Acknowledgement

1. [AndroidViewClient](https://github.com/dtmilano/AndroidViewClient)
2. [Androguard](http://code.google.com/p/androguard/)
3. [The Honeynet project](https://www.honeynet.org/)
4. [Google Summer of Code](https://summerofcode.withgoogle.com/)

## Useful links

- [DroidBot Blog Posts](http://honeynet.github.io/droidbot/)
- [droidbotApp Source Code](https://github.com/ylimit/droidbotApp)
- [How to contact the author](http://ylimit.github.io)
---

## Updated Droidbot by gaolongxi

### New Functions

1. Add a SysDataMonitor class to record the system-level data(mem, cpu, gpu)
2. Add a WaitUserLoginEvent in input_event.py to test the login page and remind the user to log on a specific device
3. Add a dump_ui_xml() function in device.py to save the xml files in the states folder just like screenshots and json files
4. Add a control folder to apply to the concurrent testing on a serial of devices

### How to use

1. make sure you have python3
2. make sure the APKPackage folder exists
3. set the devices' ip in the control/control_crawl.py
4. make sure the devices is alive, and you can restart the devices

   ```shell
   adb kill-server
   adb start-server
   ```

5. make sure you are in the control folder and run the control_crawl.py

   ```shell
    cd control
    python3 control_crawl.py
   ```

6. when there are new txt files created in the login_signal, which means that the tested app in a specific device is on the login page and you need to manually log in and send the feedback to droidbot
   
   ```shell
    python3 signal_login.py [ip]
   ```

### Operating result

1. Folder and file structure

    ```shell
    droidbot/
    ├── APKPackage/
    │ ├── xxx.apk
    │ ├── xxx.apk
    ├── control/
    │ ├── login_signal
    │ ├── output
    │   ├── device_ip1
    │   │ ├──apk1
    │   │ ├──apk2
    │   │ └── ...
    │   ├── device_ip2
    │   │ ├──apk3
    │   │ ├──apk4
    │   │ └── ...
    └── droidbot/
    ```

2. Workflow of control_crawl.py

   + distribute the app in the apk and create the  needed folders
   + run the droidbot on each device and the output for each droidbot has been redirected to a specific crawling log file in its own apk folder
