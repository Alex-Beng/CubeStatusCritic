# Cstimer framework

signal based.

## How to input time from multi sources

emit signal `time` in each source. 

data format is like this:

| source | format | meaning |
|--------|--------|---------|
| inside/outside timer, like stackmat/input/other bluetooth timer | time | [[`penalty time`, `phaseN`, `phaseN-1`, ...], ]  |
| smart cube | ["", 0, time, 0 [sol, '333']] | `""` is smart/vitual cube tag; <br> num 0 for using curScramble; <br>  time format like timer; <br> 0 for timestamp; <br>  array of solution and event tag |
| virtual cube | ["", 0, time, 0 [sol, '333', movecnt]] | compared to smart cube, one more move count |


## How to export time to file


