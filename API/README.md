# API/ — piezo SDK (not included in the repository)

Acquisition drives a **Physik Instrumente E-816** piezo controller through its
official DLL. The SDK is PI proprietary software and **cannot be
redistributed**, so it does not travel in git: install the *PI Software Suite*
(or copy the E-816 SDK) and place these three files here:

```
API/PI/E816_DLL.h
API/PI/E816_DLL_x64.dll
API/PI/E816_DLL_x64.lib
```

The application looks for the DLL at that fixed path when you press "connect
piezo" (`views/AcquisitionPanel.py`); it is not loaded before that, so without
it the app still works fully in analysis/visualization mode.
