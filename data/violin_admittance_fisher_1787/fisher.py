"""
Data from Saitis, 2012, Bridge admittance measurements of 10 preference-rated violins.
https://www.music.mcgill.ca/caml/lib/exe/fetch.php?media=publications:nantes2012_saitis.pdf
The violin is the one they refer to as D, which their testers preferred the most.
It's by Fisher, 1787.
"""

import scipy.interpolate
import data.violin_admittance_fisher_1787.fisher as fisher

def admittance(f):
  """
  Input is frequency in Hz. Output is bridge admittance as a gain factor (linear, not db).
  """
  if not hasattr(admittance,"func"):
    x = []
    y = []
    pts = admittance_list()
    for p in pts:
      xx,yy = p
      x.append(xx)
      y.append(yy)
    admittance.func = scipy.interpolate.interp1d(
            x,y,
            fill_value=(-48.6681,-28.7822),
            bounds_error=False) # default is linear
    # https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html#d-interpolation-interp1d
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d
  a = admittance.func(f)+30 # arbitrarily add a constant so values aren't so small
  return 10.0**(a/10.0)

def admittance_list():
  """
  X axis is frequency in Hz.
  Y axis is bridge admittance in db.
  """
  return [
    [  200.5,-48.6681],
    [  208.8,-45.0026],
    [  213.2,-45.6940],
    [  215.8,-46.5281],
    [  229.7,-43.1259],
    [  241.0,-40.8762],
    [  247.0,-39.2300],
    [  250.5,-38.0667],
    [  255.3,-40.6457],
    [  263.5,-36.7717],
    [  273.2,-32.8208],
    [  276.9,-36.1790],
    [  279.7,-40.2506],
    [  281.1,-42.6431],
    [  282.3,-44.6185],
    [  285.3,-43.5759],
    [  289.1,-44.5636],
    [  293.3,-44.5856],
    [  300.5,-42.5004],
    [  307.4,-41.0078],
    [  311.5,-40.2945],
    [  315.8,-40.4372],
    [  323.4,-38.9995],
    [  334.9,-37.5289],
    [  345.6,-36.4095],
    [  352.3,-36.0364],
    [  358.2,-35.3120],
    [  368.7,-34.1158],
    [  380.5,-32.2611],
    [  383.6,-33.1500],
    [  395.0,-30.1869],
    [  410.4,-25.5008],
    [  416.3,-30.0442],
    [  417.7,-34.4012],
    [  422.1,-38.4837],
    [  428.2,-35.1035],
    [  434.4,-32.0636],
    [  436.8,-31.9977],
    [  443.4,-29.5504],
    [  446.3,-29.2431],
    [  448.6,-29.5504],
    [  449.8,-30.5052],
    [  452.5,-31.3392],
    [  454.1,-31.5039],
    [  456.1,-31.2953],
    [  459.4,-30.4503],
    [  461.9,-29.3748],
    [  465.0,-30.1979],
    [  465.2,-30.9442],
    [  468.1,-31.6794],
    [  472.2,-29.8467],
    [  477.2,-27.8823],
    [  484.1,-25.0398],
    [  490.8,-22.5047],
    [  494.7,-21.7694],
    [  499.0,-21.9230],
    [  504.3,-21.9011],
    [  510.5,-24.6118],
    [  517.0,-27.7286],
    [  523.0,-30.9771],
    [  527.4,-33.4793],
    [  531.2,-34.5767],
    [  535.3,-34.7962],
    [  538.3,-34.4121],
    [  544.0,-32.2172],
    [  549.7,-30.0442],
    [  553.1,-28.5517],
    [  556.8,-28.1786],
    [  559.8,-28.5846],
    [  563.9,-29.8577],
    [  565.4,-31.2734],
    [  568.6,-30.8893],
    [  574.6,-32.6342],
    [  580.6,-34.3134],
    [  588.1,-35.8827],
    [  597.6,-38.1874],
    [  601.9,-39.6470],
    [  610.9,-41.8858],
    [  618.4,-40.2396],
    [  623.5,-42.9833],
    [  628.7,-44.9038],
    [  641.7,-47.4828],
    [  647.1,-48.0206],
    [  653.5,-48.2072],
    [  664.4,-46.9012],
    [  670.7,-46.5720],
    [  675.6,-47.0000],
    [  681.6,-47.9328],
    [  685.3,-48.3498],
    [  690.3,-47.8999],
    [  702.2,-44.8160],
    [  714.7,-42.5553],
    [  719.9,-41.9956],
    [  727.5,-41.8200],
    [  748.3,-39.9433],
    [  769.2,-37.8033],
    [  775.6,-37.8143],
    [  781.2,-38.2532],
    [  804.0,-36.8046],
    [  832.9,-34.3024],
    [  857.2,-31.7672],
    [  879.3,-29.0126],
    [  889.6,-30.3954],
    [  900.9,-28.4420],
    [  911.0,-26.6641],
    [  922.1,-30.6478],
    [  927.2,-33.8744],
    [  932.3,-35.2681],
    [  940.1,-35.7291],
    [  946.4,-35.0706],
    [  954.8,-32.8428],
    [  957.4,-30.9771],
    [  964.3,-29.4187],
    [  969.7,-29.1114],
    [  973.9,-29.4736],
    [  977.7,-31.8331],
    [  981.5,-33.5671],
    [  988.0,-34.1817],
    [  996.8,-33.6439],
    [ 1010.1,-31.0100],
    [ 1015.2,-30.7247],
    [ 1021.9,-31.0868],
    [ 1031.6,-34.0280],
    [ 1041.3,-36.2888],
    [ 1052.3,-36.7936],
    [ 1060.5,-36.3327],
    [ 1082.4,-33.1391],
    [ 1102.3,-30.5820],
    [ 1117.0,-28.4859],
    [ 1123.8,-26.5214],
    [ 1126.9,-25.8849],
    [ 1130.7,-25.5117],
    [ 1137.6,-25.8629],
    [ 1142.0,-26.9165],
    [ 1148.9,-28.0030],
    [ 1155.3,-28.2334],
    [ 1165.6,-27.7944],
    [ 1173.3,-28.1237],
    [ 1183.7,-30.6588],
    [ 1189.0,-32.2062],
    [ 1192.3,-33.7427],
    [ 1201.5,-34.3902],
    [ 1212.2,-33.4683],
    [ 1220.9,-33.3256],
    [ 1225.7,-33.4683],
    [ 1236.6,-33.2817],
    [ 1244.8,-33.5781],
    [ 1251.7,-35.0706],
    [ 1262.1,-36.6839],
    [ 1266.3,-36.9582],
    [ 1278.2,-36.1900],
    [ 1283.2,-37.7484],
    [ 1286.7,-42.8296],
    [ 1292.4,-36.8485],
    [ 1297.5,-34.4890],
    [ 1314.0,-31.9758],
    [ 1322.1,-34.0500],
    [ 1325.7,-36.3985],
    [ 1332.3,-37.9679],
    [ 1338.2,-38.8678],
    [ 1355.3,-36.3437],
    [ 1369.7,-34.1048],
    [ 1381.1,-33.6878],
    [ 1404.9,-33.9402],
    [ 1432.4,-32.0087],
    [ 1463.6,-28.2115],
    [ 1488.0,-26.2470],
    [ 1497.9,-27.2238],
    [ 1509.6,-28.0249],
    [ 1528.0,-26.9714],
    [ 1538.2,-27.4433],
    [ 1553.6,-29.2870],
    [ 1564.8,-30.7356],
    [ 1590.9,-31.1417],
    [ 1618.4,-28.8041],
    [ 1637.3,-31.3722],
    [ 1652.8,-30.8564],
    [ 1669.3,-31.2405],
    [ 1697.2,-28.1017],
    [ 1708.5,-27.5311],
    [ 1721.7,-28.1566],
    [ 1726.5,-29.2650],
    [ 1741.8,-29.3309],
    [ 1742.8,-30.5271],
    [ 1760.2,-31.5478],
    [ 1769.0,-32.3928],
    [ 1785.7,-31.0649],
    [ 1790.6,-30.3296],
    [ 1806.5,-30.2198],
    [ 1828.6,-26.6750],
    [ 1863.3,-23.5473],
    [ 1881.9,-25.0508],
    [ 1889.2,-28.3212],
    [ 1905.0,-40.7115],
    [ 1917.6,-35.2791],
    [ 1929.3,-36.2668],
    [ 1963.8,-31.5039],
    [ 1990.0,-28.1017],
    [ 2021.0,-24.1509],
    [ 2055.9,-21.7913],
    [ 2081.1,-24.0082],
    [ 2096.1,-26.3568],
    [ 2122.9,-28.0578],
    [ 2141.7,-27.6298],
    [ 2181.1,-26.1812],
    [ 2206.6,-26.0934],
    [ 2229.9,-24.0302],
    [ 2253.4,-20.6829],
    [ 2268.4,-19.9476],
    [ 2287.3,-19.6842],
    [ 2297.4,-20.0135],
    [ 2314.0,-21.8352],
    [ 2337.1,-24.0960],
    [ 2355.3,-21.9121],
    [ 2380.1,-19.5855],
    [ 2400.0,-21.0561],
    [ 2409.3,-22.0218],
    [ 2426.6,-22.5815],
    [ 2452.2,-21.9450],
    [ 2468.6,-22.8778],
    [ 2480.9,-25.2593],
    [ 2501.5,-24.9081],
    [ 2515.4,-25.9398],
    [ 2533.5,-27.8054],
    [ 2570.2,-24.8642],
    [ 2598.7,-32.8757],
    [ 2616.0,-30.7686],
    [ 2633.4,-43.8942],
    [ 2645.1,-31.9977],
    [ 2668.6,-42.7199],
    [ 2701.2,-22.8778],
    [ 2711.7,-22.0657],
    [ 2729.7,-22.7681],
    [ 2750.9,-21.4292],
    [ 2780.0,-23.1632],
    [ 2814.0,-23.8655],
    [ 2859.4,-22.8120],
    [ 2886.4,-24.1180],
    [ 2923.4,-26.8726],
    [ 2959.1,-24.6667],
    [ 2978.8,-23.6680],
    [ 3003.6,-23.2510],
    [ 3026.9,-24.3155],
    [ 3048.7,-25.6544],
    [ 3079.2,-25.5556],
    [ 3104.8,-27.0043],
    [ 3130.7,-29.2321],
    [ 3170.7,-26.3239],
    [ 3195.4,-27.1579],
    [ 3238.0,-22.9986],
    [ 3266.8,-21.1329],
    [ 3312.2,-19.2782],
    [ 3358.3,-21.9889],
    [ 3384.4,-22.0218],
    [ 3416.3,-24.0411],
    [ 3456.2,-24.9411],
    [ 3479.2,-26.2031],
    [ 3517.9,-24.4253],
    [ 3535.4,-24.1948],
    [ 3612.4,-24.6447],
    [ 3644.5,-25.8629],
    [ 3672.8,-27.6298],
    [ 3711.6,-28.6066],
    [ 3742.5,-27.9810],
    [ 3757.0,-28.2554],
    [ 3790.3,-33.9951],
    [ 3815.6,-23.0973],
    [ 3858.0,-23.9204],
    [ 3905.1,-22.8888],
    [ 3952.9,-23.0644],
    [ 4001.3,-25.2374],
    [ 4045.7,-28.3761],
    [ 4083.9,-33.9402],
    [ 4108.8,-25.7971],
    [ 4131.6,-19.8598],
    [ 4163.7,-22.0657],
    [ 4249.7,-24.6118],
    [ 4378.4,-26.1044],
    [ 4412.4,-25.1276],
    [ 4446.7,-24.9081],
    [ 4491.1,-25.1166],
    [ 4526.0,-26.2361],
    [ 4571.3,-26.8836],
    [ 4658.0,-27.4652],
    [ 4709.7,-27.6079],
    [ 4770.0,-27.1579],
    [ 4809.7,-27.5969],
    [ 4839.0,-27.4872],
    [ 4879.3,-27.9371],
    [ 4955.4,-28.3103],
    [ 4991.1,-28.7822]
  ]

