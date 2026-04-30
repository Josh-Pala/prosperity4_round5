# PCA + lead-lag follow-up

## PCA on 10-tick returns

| pc   |   var_explained |   n_families_in_top10 | families_in_top10                                                             | top_symbols                                                                                                                                                                           |
|:-----|----------------:|----------------------:|:------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PC1  |          0.0571 |                     7 | GALAXY_SOUNDS,MICROCHIP,PANEL,ROBOT,SNACKPACK,TRANSLATOR,UV_VISOR             | SNACKPACK_STRAWBERRY,SNACKPACK_RASPBERRY,SNACKPACK_PISTACHIO,MICROCHIP_SQUARE,TRANSLATOR_VOID_BLUE,ROBOT_VACUUMING,GALAXY_SOUNDS_SOLAR_FLAMES,MICROCHIP_CIRCLE,UV_VISOR_RED,PANEL_4X4 |
| PC2  |          0.0416 |                     5 | MICROCHIP,PANEL,PEBBLES,SLEEP_POD,SNACKPACK                                   | PEBBLES_XL,PEBBLES_M,PEBBLES_XS,PEBBLES_S,PEBBLES_L,SNACKPACK_VANILLA,SNACKPACK_CHOCOLATE,MICROCHIP_SQUARE,PANEL_1X4,SLEEP_POD_POLYESTER                                              |
| PC3  |          0.0396 |                     5 | MICROCHIP,OXYGEN_SHAKE,PEBBLES,SNACKPACK,TRANSLATOR                           | SNACKPACK_CHOCOLATE,SNACKPACK_VANILLA,PEBBLES_XL,PEBBLES_XS,PEBBLES_L,PEBBLES_M,PEBBLES_S,OXYGEN_SHAKE_MINT,TRANSLATOR_VOID_BLUE,MICROCHIP_TRIANGLE                                   |
| PC4  |          0.0236 |                     7 | MICROCHIP,OXYGEN_SHAKE,PEBBLES,ROBOT,SLEEP_POD,TRANSLATOR,UV_VISOR            | SLEEP_POD_POLYESTER,PEBBLES_L,MICROCHIP_CIRCLE,PEBBLES_S,SLEEP_POD_NYLON,UV_VISOR_MAGENTA,TRANSLATOR_SPACE_GRAY,OXYGEN_SHAKE_CHOCOLATE,TRANSLATOR_GRAPHITE_MIST,ROBOT_DISHES          |
| PC5  |          0.0230 |                     7 | GALAXY_SOUNDS,MICROCHIP,OXYGEN_SHAKE,PEBBLES,ROBOT,SLEEP_POD,UV_VISOR         | UV_VISOR_RED,OXYGEN_SHAKE_MINT,UV_VISOR_ORANGE,ROBOT_IRONING,PEBBLES_XS,SLEEP_POD_COTTON,OXYGEN_SHAKE_GARLIC,ROBOT_VACUUMING,MICROCHIP_RECTANGLE,GALAXY_SOUNDS_DARK_MATTER            |
| PC6  |          0.0229 |                     8 | GALAXY_SOUNDS,MICROCHIP,OXYGEN_SHAKE,PANEL,PEBBLES,ROBOT,SLEEP_POD,TRANSLATOR | PANEL_2X2,PANEL_1X4,GALAXY_SOUNDS_SOLAR_FLAMES,SLEEP_POD_SUEDE,MICROCHIP_OVAL,PEBBLES_S,TRANSLATOR_ECLIPSE_CHARCOAL,GALAXY_SOUNDS_SOLAR_WINDS,OXYGEN_SHAKE_CHOCOLATE,ROBOT_DISHES     |

## Lead-lag: top 40 cross-family pairs
Negative `best_lag` = `a` leads `b`.  Positive = `b` leads `a`.

| a                             | b                        | family_a      | family_b   |   corr_lag0 |   best_lag |   corr_at_best_lag |   improvement_vs_lag0 |
|:------------------------------|:-------------------------|:--------------|:-----------|------------:|-----------:|-------------------:|----------------------:|
| MICROCHIP_RECTANGLE           | UV_VISOR_RED             | MICROCHIP     | UV_VISOR   |      0.0323 |         -6 |             0.0521 |                0.0199 |
| ROBOT_VACUUMING               | SNACKPACK_PISTACHIO      | ROBOT         | SNACKPACK  |      0.0353 |         -5 |             0.0523 |                0.0169 |
| GALAXY_SOUNDS_DARK_MATTER     | SLEEP_POD_COTTON         | GALAXY_SOUNDS | SLEEP_POD  |      0.0337 |         13 |             0.0456 |                0.0119 |
| PEBBLES_L                     | UV_VISOR_MAGENTA         | PEBBLES       | UV_VISOR   |      0.0313 |          4 |             0.0417 |                0.0104 |
| MICROCHIP_RECTANGLE           | TRANSLATOR_GRAPHITE_MIST | MICROCHIP     | TRANSLATOR |     -0.0366 |         -4 |            -0.0464 |                0.0097 |
| GALAXY_SOUNDS_BLACK_HOLES     | TRANSLATOR_ASTRO_BLACK   | GALAXY_SOUNDS | TRANSLATOR |     -0.0380 |          3 |            -0.0452 |                0.0072 |
| MICROCHIP_RECTANGLE           | PEBBLES_S                | MICROCHIP     | PEBBLES    |     -0.0377 |          4 |            -0.0442 |                0.0065 |
| MICROCHIP_SQUARE              | SNACKPACK_PISTACHIO      | MICROCHIP     | SNACKPACK  |     -0.0421 |         -3 |            -0.0482 |                0.0062 |
| OXYGEN_SHAKE_CHOCOLATE        | SLEEP_POD_POLYESTER      | OXYGEN_SHAKE  | SLEEP_POD  |     -0.0398 |          4 |            -0.0457 |                0.0059 |
| PEBBLES_M                     | SNACKPACK_CHOCOLATE      | PEBBLES       | SNACKPACK  |     -0.0355 |         -2 |            -0.0404 |                0.0049 |
| GALAXY_SOUNDS_SOLAR_FLAMES    | SLEEP_POD_SUEDE          | GALAXY_SOUNDS | SLEEP_POD  |      0.0452 |         -4 |             0.0498 |                0.0046 |
| PANEL_1X2                     | SLEEP_POD_COTTON         | PANEL         | SLEEP_POD  |      0.0315 |         -4 |             0.0357 |                0.0042 |
| PEBBLES_M                     | SNACKPACK_VANILLA        | PEBBLES       | SNACKPACK  |      0.0338 |         -2 |             0.0378 |                0.0040 |
| MICROCHIP_TRIANGLE            | TRANSLATOR_ASTRO_BLACK   | MICROCHIP     | TRANSLATOR |      0.0401 |          2 |             0.0437 |                0.0036 |
| OXYGEN_SHAKE_MINT             | ROBOT_IRONING            | OXYGEN_SHAKE  | ROBOT      |     -0.0394 |          1 |            -0.0418 |                0.0024 |
| GALAXY_SOUNDS_PLANETARY_RINGS | ROBOT_LAUNDRY            | GALAXY_SOUNDS | ROBOT      |      0.0389 |         -2 |             0.0408 |                0.0018 |
| MICROCHIP_CIRCLE              | PEBBLES_S                | MICROCHIP     | PEBBLES    |     -0.0408 |         -1 |            -0.0425 |                0.0017 |
| PEBBLES_S                     | TRANSLATOR_VOID_BLUE     | PEBBLES       | TRANSLATOR |     -0.0416 |          2 |            -0.0431 |                0.0015 |
| OXYGEN_SHAKE_MINT             | SNACKPACK_CHOCOLATE      | OXYGEN_SHAKE  | SNACKPACK  |     -0.0327 |         -1 |            -0.0341 |                0.0014 |
| OXYGEN_SHAKE_GARLIC           | SLEEP_POD_LAMB_WOOL      | OXYGEN_SHAKE  | SLEEP_POD  |     -0.0417 |         -1 |            -0.0430 |                0.0013 |