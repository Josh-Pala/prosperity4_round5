# Cross-family EDA — Round 5

Symbols loaded (non-constant): 50
Cross-family pair count: 1125

## Top 25 cross-family pairs by |corr(10-tick returns)|

| a                             | b                        | family_a      | family_b   |   corr_ret10 |   corr_ret1 |   beta_a_on_b |   resid_sd_a_on_b |   half_life_a_on_b |
|:------------------------------|:-------------------------|:--------------|:-----------|-------------:|------------:|--------------:|------------------:|-------------------:|
| PANEL_1X4                     | PEBBLES_S                | PANEL         | PEBBLES    |      -0.0564 |     -0.0090 |        0.5912 |          673.0171 |          3772.3520 |
| SLEEP_POD_POLYESTER           | UV_VISOR_AMBER           | SLEEP_POD     | UV_VISOR   |       0.0522 |      0.0065 |       -0.9226 |          331.0176 |           817.8740 |
| MICROCHIP_SQUARE              | SNACKPACK_RASPBERRY      | MICROCHIP     | SNACKPACK  |       0.0475 |     -0.0007 |        0.3941 |         1829.0284 |          5377.2681 |
| GALAXY_SOUNDS_SOLAR_FLAMES    | SLEEP_POD_SUEDE          | GALAXY_SOUNDS | SLEEP_POD  |       0.0452 |      0.0076 |       -0.0610 |          446.7860 |          1647.5156 |
| ROBOT_DISHES                  | SLEEP_POD_POLYESTER      | ROBOT         | SLEEP_POD  |      -0.0440 |     -0.0046 |        0.4220 |          373.6750 |           606.4508 |
| MICROCHIP_SQUARE              | SNACKPACK_PISTACHIO      | MICROCHIP     | SNACKPACK  |      -0.0421 |      0.0007 |       -6.6953 |         1331.8994 |          1696.9552 |
| GALAXY_SOUNDS_PLANETARY_RINGS | PEBBLES_M                | GALAXY_SOUNDS | PEBBLES    |      -0.0419 |      0.0029 |        0.1969 |          753.7612 |          8228.8791 |
| OXYGEN_SHAKE_GARLIC           | SLEEP_POD_LAMB_WOOL      | OXYGEN_SHAKE  | SLEEP_POD  |      -0.0417 |     -0.0023 |        0.3932 |          939.4058 |          9382.2428 |
| PEBBLES_S                     | TRANSLATOR_VOID_BLUE     | PEBBLES       | TRANSLATOR |      -0.0416 |     -0.0052 |       -0.9100 |          645.3662 |          1790.9113 |
| MICROCHIP_SQUARE              | SNACKPACK_STRAWBERRY     | MICROCHIP     | SNACKPACK  |      -0.0414 |      0.0035 |        4.0410 |         1091.4522 |          1097.9904 |
| PEBBLES_XS                    | ROBOT_DISHES             | PEBBLES       | ROBOT      |       0.0410 |      0.0012 |       -2.0292 |          908.5044 |           667.9219 |
| MICROCHIP_CIRCLE              | PEBBLES_S                | MICROCHIP     | PEBBLES    |      -0.0408 |      0.0034 |       -0.3382 |          451.8065 |          2187.2518 |
| MICROCHIP_TRIANGLE            | TRANSLATOR_ASTRO_BLACK   | MICROCHIP     | TRANSLATOR |       0.0401 |     -0.0009 |        0.5207 |          793.4009 |          5552.7394 |
| PEBBLES_XL                    | TRANSLATOR_VOID_BLUE     | PEBBLES       | TRANSLATOR |       0.0400 |      0.0093 |        2.4823 |         1043.3654 |           927.0940 |
| OXYGEN_SHAKE_CHOCOLATE        | SLEEP_POD_POLYESTER      | OXYGEN_SHAKE  | SLEEP_POD  |      -0.0398 |      0.0062 |        0.1880 |          529.6219 |          3838.8683 |
| OXYGEN_SHAKE_MINT             | ROBOT_IRONING            | OXYGEN_SHAKE  | ROBOT      |      -0.0394 |     -0.0007 |        0.2949 |          454.4125 |          2924.3788 |
| SNACKPACK_RASPBERRY           | TRANSLATOR_VOID_BLUE     | SNACKPACK     | TRANSLATOR |       0.0393 |      0.0049 |       -0.0047 |          169.7924 |           624.0695 |
| GALAXY_SOUNDS_PLANETARY_RINGS | ROBOT_LAUNDRY            | GALAXY_SOUNDS | ROBOT      |       0.0389 |      0.0075 |       -0.0639 |          764.8292 |          8678.4773 |
| GALAXY_SOUNDS_BLACK_HOLES     | TRANSLATOR_ASTRO_BLACK   | GALAXY_SOUNDS | TRANSLATOR |      -0.0380 |      0.0128 |       -0.8040 |          873.8330 |          7997.8561 |
| MICROCHIP_CIRCLE              | UV_VISOR_YELLOW          | MICROCHIP     | UV_VISOR   |      -0.0380 |     -0.0016 |       -0.5716 |          362.9114 |          1633.7836 |
| OXYGEN_SHAKE_MINT             | SNACKPACK_VANILLA        | OXYGEN_SHAKE  | SNACKPACK  |       0.0378 |      0.0280 |       -0.7460 |          490.3706 |          2864.7268 |
| MICROCHIP_RECTANGLE           | PEBBLES_S                | MICROCHIP     | PEBBLES    |      -0.0377 |     -0.0043 |        0.4789 |          637.4132 |          2374.8787 |
| MICROCHIP_RECTANGLE           | TRANSLATOR_GRAPHITE_MIST | MICROCHIP     | TRANSLATOR |      -0.0366 |     -0.0035 |       -0.7192 |          660.6433 |          2221.0013 |
| PANEL_1X4                     | PEBBLES_XL               | PANEL         | PEBBLES    |       0.0361 |      0.0022 |       -0.1667 |          779.6731 |          7472.8534 |
| SLEEP_POD_POLYESTER           | UV_VISOR_MAGENTA         | SLEEP_POD     | UV_VISOR   |       0.0360 |      0.0144 |        1.3673 |          501.7832 |           949.0787 |

## Family-vs-family map (mean |corr| of 10-tick returns)

| pair_family                |   mean_abs_r10 |   max_abs_r10 |       n |
|:---------------------------|---------------:|--------------:|--------:|
| MICROCHIP | SNACKPACK      |         0.0194 |        0.0475 | 25.0000 |
| MICROCHIP | PEBBLES        |         0.0176 |        0.0408 | 25.0000 |
| GALAXY_SOUNDS | SNACKPACK  |         0.0156 |        0.0311 | 25.0000 |
| MICROCHIP | UV_VISOR       |         0.0156 |        0.0380 | 25.0000 |
| PEBBLES | TRANSLATOR       |         0.0156 |        0.0416 | 25.0000 |
| PEBBLES | SLEEP_POD        |         0.0154 |        0.0308 | 25.0000 |
| SNACKPACK | TRANSLATOR     |         0.0146 |        0.0393 | 25.0000 |
| PEBBLES | SNACKPACK        |         0.0146 |        0.0355 | 25.0000 |
| ROBOT | SNACKPACK          |         0.0142 |        0.0353 | 25.0000 |
| SLEEP_POD | UV_VISOR       |         0.0141 |        0.0522 | 25.0000 |
| GALAXY_SOUNDS | PEBBLES    |         0.0141 |        0.0419 | 25.0000 |
| MICROCHIP | TRANSLATOR     |         0.0139 |        0.0401 | 25.0000 |
| MICROCHIP | PANEL          |         0.0138 |        0.0293 | 25.0000 |
| SLEEP_POD | TRANSLATOR     |         0.0136 |        0.0317 | 25.0000 |
| OXYGEN_SHAKE | PANEL       |         0.0136 |        0.0338 | 25.0000 |
| MICROCHIP | ROBOT          |         0.0136 |        0.0298 | 25.0000 |
| OXYGEN_SHAKE | SLEEP_POD   |         0.0134 |        0.0417 | 25.0000 |
| GALAXY_SOUNDS | SLEEP_POD  |         0.0133 |        0.0452 | 25.0000 |
| PANEL | SLEEP_POD          |         0.0131 |        0.0315 | 25.0000 |
| PANEL | PEBBLES            |         0.0131 |        0.0564 | 25.0000 |
| OXYGEN_SHAKE | TRANSLATOR  |         0.0131 |        0.0318 | 25.0000 |
| PEBBLES | UV_VISOR         |         0.0127 |        0.0313 | 25.0000 |
| GALAXY_SOUNDS | TRANSLATOR |         0.0123 |        0.0380 | 25.0000 |
| MICROCHIP | OXYGEN_SHAKE   |         0.0122 |        0.0232 | 25.0000 |
| GALAXY_SOUNDS | MICROCHIP  |         0.0121 |        0.0313 | 25.0000 |
| ROBOT | SLEEP_POD          |         0.0118 |        0.0440 | 25.0000 |
| ROBOT | UV_VISOR           |         0.0116 |        0.0358 | 25.0000 |
| OXYGEN_SHAKE | PEBBLES     |         0.0114 |        0.0267 | 25.0000 |
| SNACKPACK | UV_VISOR       |         0.0114 |        0.0303 | 25.0000 |
| GALAXY_SOUNDS | UV_VISOR   |         0.0112 |        0.0308 | 25.0000 |