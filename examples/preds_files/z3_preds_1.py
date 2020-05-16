from z3 import *

x_1 = Real('x_1')
x_2 = Real('x_2')
x_3 = Real('x_3')
x_4 = Real('x_4')
s = Optimize()
set_option(rational_to_decimal=True)
c_1 = Bool('c_1')
s.add(c_1 == And(-0.9373384426740606*x_1 + 0.20802145704967556*x_2 <= -0.6460158846771265, 1.0*x_1 <= 1.0, -1.0*x_1 <= -0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= -0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= -0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= -0.0))
s.add_soft(c_1)
c_2 = Bool('c_2')
s.add(c_2 == And(-0.9283442885951491*x_1 + 0.2190951789549774*x_2 <= -0.25322422996621263, 1.0*x_1 <= 1.0, -1.0*x_1 <= -0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= -0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= -0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= -0.0))
s.add_soft(c_2)
c_3 = Bool('c_3')
s.add(c_3 == And(-0.918897115832545*x_1 + 0.22954326103129677*x_2 <= 0.1356840999989446, 1.0*x_1 <= 1.0, -1.0*x_1 <= -0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= -0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= -0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= -0.0))
s.add_soft(c_3)
c_4 = Bool('c_4')
s.add(c_4 == And(-0.9090229039890286*x_1 + 0.23938123866151031*x_2 <= 0.5205238780918346, 1.0*x_1 <= 1.0, -1.0*x_1 <= -0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= -0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= -0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= -0.0))
s.add_soft(c_4)
c_5 = Bool('c_5')
s.add(c_5 == And(-0.8987470352257187*x_1 + 0.2486245365918096*x_2 <= 0.9011206568533989, 1.0*x_1 <= 1.0, -1.0*x_1 <= -0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= -0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= -0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= -0.0))
s.add_soft(c_5)
c_6 = Bool('c_6')
s.add(c_6 == And(-0.88809422517863*x_1 + 0.25728849855651836*x_2 <= 1.2773105423004338, 1.0*x_1 <= 1.0, -1.0*x_1 <= -0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= -0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= -0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= -0.0))
s.add_soft(c_6)
c_7 = Bool('c_7')
s.add(c_7 == And(-0.8770885456037522*x_1 + 0.2653883692643328*x_2 <= 1.648939912755285, 1.0*x_1 <= 1.0, -1.0*x_1 <= -0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= -0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= -0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= -0.0))
s.add_soft(c_7)
c_8 = Bool('c_8')
s.add(c_8 == And(-0.8657534287312288*x_1 + 0.272939282802461*x_2 <= 2.0158651471595306, 1.0*x_1 <= 1.0, -1.0*x_1 <= -0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= -0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= -0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= -0.0))
s.add_soft(c_8)
c_9 = Bool('c_9')
s.add(c_9 == And(-0.8541116720419177*x_1 + 0.27995625720026013*x_2 <= 2.3779523568323704, 1.0*x_1 <= 1.0, -1.0*x_1 <= -0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= -0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= -0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= -0.0))
s.add_soft(c_9)
c_10 = Bool('c_10')
s.add(c_10 == And(-0.8421854431028581*x_1 + 0.286454183213154*x_2 <= 2.7350771205056166, 1.0*x_1 <= 1.0, -1.0*x_1 <= -0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= -0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= -0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= -0.0))
s.add_soft(c_10)
c_11 = Bool('c_11')
s.add(c_11 == And(-0.8299963049639894*x_1 + 0.2924478200791091*x_2 <= 2.6661333442457096, 1.0*x_1 <= 1.0, -1.0*x_1 <= 0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= 0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= 0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= 0.0, -1.0*x_3 <= 5.714999999999995e-07, 1.0*x_3 <= 2.8000000000000003e-05, -1.0*x_4 <= 2.2e-05, 1.0*x_4 <= 0.000978, -0.8421854431028581*x_1 + 0.286454183213154*x_2 <= 1008.7350771205056, 0.8421854431028581*x_1 + -0.286454183213154*x_2 <= 991.2649228794944, -0.5729085324850244*x_1 + -0.2965583040292779*x_2 <= 1003.1562920701859, 0.5729085324850244*x_1 + 0.2965583040292779*x_2 <= 996.8437079298141))
s.add_soft(c_11)
c_12 = Bool('c_12')
s.add(c_12 == And(-0.817565143034*x_1 + 0.2979517780098138*x_2 <= 2.5923700067886966, 1.0*x_1 <= 1.0, -1.0*x_1 <= 0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= 0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= 0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= 0.0, -1.0*x_3 <= 5.714999999999995e-07, 1.0*x_3 <= 2.8000000000000003e-05, -1.0*x_4 <= 2.2e-05, 1.0*x_4 <= 0.000978, -0.8421854431028581*x_1 + 0.286454183213154*x_2 <= 1008.7350771205056, 0.8421854431028581*x_1 + -0.286454183213154*x_2 <= 991.2649228794944, -0.5729085324850244*x_1 + -0.2965583040292779*x_2 <= 1003.1562920701859, 0.5729085324850244*x_1 + 0.2965583040292779*x_2 <= 996.8437079298141))
s.add_soft(c_12)
c_13 = Bool('c_13')
s.add(c_13 == And(-0.8049123112891305*x_1 + 0.3029805265254552*x_2 <= 2.514040998755952, 1.0*x_1 <= 1.0, -1.0*x_1 <= 0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= 0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= 0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= 0.0, -1.0*x_3 <= 5.714999999999995e-07, 1.0*x_3 <= 2.8000000000000003e-05, -1.0*x_4 <= 2.2e-05, 1.0*x_4 <= 0.000978, -0.8421854431028581*x_1 + 0.286454183213154*x_2 <= 1008.7350771205056, 0.8421854431028581*x_1 + -0.286454183213154*x_2 <= 991.2649228794944, -0.5729085324850244*x_1 + -0.2965583040292779*x_2 <= 1003.1562920701859, 0.5729085324850244*x_1 + 0.2965583040292779*x_2 <= 996.8437079298141))
s.add_soft(c_13)
c_14 = Bool('c_14')
s.add(c_14 == And(-0.7920574752645262*x_1 + 0.30754837167118776*x_2 <= 2.43139347103177, 1.0*x_1 <= 1.0, -1.0*x_1 <= 0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= 0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= 0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= 0.0, -1.0*x_3 <= 5.714999999999995e-07, 1.0*x_3 <= 2.8000000000000003e-05, -1.0*x_4 <= 2.2e-05, 1.0*x_4 <= 0.000978, -0.8421854431028581*x_1 + 0.286454183213154*x_2 <= 1008.7350771205056, 0.8421854431028581*x_1 + -0.286454183213154*x_2 <= 991.2649228794944, -0.5729085324850244*x_1 + -0.2965583040292779*x_2 <= 1003.1562920701859, 0.5729085324850244*x_1 + 0.2965583040292779*x_2 <= 996.8437079298141))
s.add_soft(c_14)
c_15 = Bool('c_15')
s.add(c_15 == And(-0.7790197637918801*x_1 + 0.3116694633592647*x_2 <= 2.3446691576150833, 1.0*x_1 <= 1.0, -1.0*x_1 <= 0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= 0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= 0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= 0.0, -1.0*x_3 <= 5.714999999999995e-07, 1.0*x_3 <= 2.8000000000000003e-05, -1.0*x_4 <= 2.2e-05, 1.0*x_4 <= 0.000978, -0.8421854431028581*x_1 + 0.286454183213154*x_2 <= 1008.7350771205056, 0.8421854431028581*x_1 + -0.286454183213154*x_2 <= 991.2649228794944, -0.5729085324850244*x_1 + -0.2965583040292779*x_2 <= 1003.1562920701859, 0.5729085324850244*x_1 + 0.2965583040292779*x_2 <= 996.8437079298141))
s.add_soft(c_15)
c_16 = Bool('c_16')
s.add(c_16 == And(-0.7658176930008399*x_1 + 0.3153577814982387*x_2 <= 2.2541036552757454, 1.0*x_1 <= 1.0, -1.0*x_1 <= 0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= 0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= 0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= 0.0, -1.0*x_3 <= 5.714999999999995e-07, 1.0*x_3 <= 2.8000000000000003e-05, -1.0*x_4 <= 2.2e-05, 1.0*x_4 <= 0.000978, -0.8421854431028581*x_1 + 0.286454183213154*x_2 <= 1008.7350771205056, 0.8421854431028581*x_1 + -0.286454183213154*x_2 <= 991.2649228794944, -0.5729085324850244*x_1 + -0.2965583040292779*x_2 <= 1003.1562920701859, 0.5729085324850244*x_1 + 0.2965583040292779*x_2 <= 996.8437079298141))
s.add_soft(c_16)
c_17 = Bool('c_17')
s.add(c_17 == And(-0.7524692138347874*x_1 + 0.3186271349237317*x_2 <= 2.1599268187027327, 1.0*x_1 <= 1.0, -1.0*x_1 <= 0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= 0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= 0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= 0.0, -1.0*x_3 <= 5.714999999999995e-07, 1.0*x_3 <= 2.8000000000000003e-05, -1.0*x_4 <= 2.2e-05, 1.0*x_4 <= 0.000978, -0.8421854431028581*x_1 + 0.286454183213154*x_2 <= 1008.7350771205056, 0.8421854431028581*x_1 + -0.286454183213154*x_2 <= 991.2649228794944, -0.5729085324850244*x_1 + -0.2965583040292779*x_2 <= 1003.1562920701859, 0.5729085324850244*x_1 + 0.2965583040292779*x_2 <= 996.8437079298141))
s.add_soft(c_17)
c_18 = Bool('c_18')
s.add(c_18 == And(-0.738991637241479*x_1 + 0.3214911471525826*x_2 <= 2.062362048075812, 1.0*x_1 <= 1.0, -1.0*x_1 <= 0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= 0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= 0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= 0.0, -1.0*x_3 <= 5.714999999999995e-07, 1.0*x_3 <= 2.8000000000000003e-05, -1.0*x_4 <= 2.2e-05, 1.0*x_4 <= 0.000978, -0.8421854431028581*x_1 + 0.286454183213154*x_2 <= 1008.7350771205056, 0.8421854431028581*x_1 + -0.286454183213154*x_2 <= 991.2649228794944, -0.5729085324850244*x_1 + -0.2965583040292779*x_2 <= 1003.1562920701859, 0.5729085324850244*x_1 + 0.2965583040292779*x_2 <= 996.8437079298141))
s.add_soft(c_18)
c_19 = Bool('c_19')
s.add(c_19 == And(-0.7254017680455608*x_1 + 0.32396326452495283*x_2 <= 1.9616274656042547, 1.0*x_1 <= 1.0, -1.0*x_1 <= 0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= 0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= 0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= 0.0, -1.0*x_3 <= 5.714999999999995e-07, 1.0*x_3 <= 2.8000000000000003e-05, -1.0*x_4 <= 2.2e-05, 1.0*x_4 <= 0.000978, -0.8421854431028581*x_1 + 0.286454183213154*x_2 <= 1008.7350771205056, 0.8421854431028581*x_1 + -0.286454183213154*x_2 <= 991.2649228794944, -0.5729085324850244*x_1 + -0.2965583040292779*x_2 <= 1003.1562920701859, 0.5729085324850244*x_1 + 0.2965583040292779*x_2 <= 996.8437079298141))
s.add_soft(c_19)
c_20 = Bool('c_20')
s.add(c_20 == And(-0.7117158274388188*x_1 + 0.3260567433312239*x_2 <= 1.8579351880792645, 1.0*x_1 <= 1.0, -1.0*x_1 <= 0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= 0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= 0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= 0.0, -1.0*x_3 <= 5.714999999999995e-07, 1.0*x_3 <= 2.8000000000000003e-05, -1.0*x_4 <= 2.2e-05, 1.0*x_4 <= 0.000978, -0.8421854431028581*x_1 + 0.286454183213154*x_2 <= 1008.7350771205056, 0.8421854431028581*x_1 + -0.286454183213154*x_2 <= 991.2649228794944, -0.5729085324850244*x_1 + -0.2965583040292779*x_2 <= 1003.1562920701859, 0.5729085324850244*x_1 + 0.2965583040292779*x_2 <= 996.8437079298141))
s.add_soft(c_20)
c_21 = Bool('c_21')
s.add(c_21 == And(-0.6979494822926586*x_1 + 0.32778464797499585*x_2 <= 1.7514915639662476, 1.0*x_1 <= 1.0, -1.0*x_1 <= 0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= 0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= 0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= 0.0, -1.0*x_3 <= 5.714999999999995e-07, 1.0*x_3 <= 2.8000000000000003e-05, -1.0*x_4 <= 2.2e-05, 1.0*x_4 <= 0.000978, -0.8421854431028581*x_1 + 0.286454183213154*x_2 <= 1008.7350771205056, 0.8421854431028581*x_1 + -0.286454183213154*x_2 <= 991.2649228794944, -0.5729085324850244*x_1 + -0.2965583040292779*x_2 <= 1003.1562920701859, 0.5729085324850244*x_1 + 0.2965583040292779*x_2 <= 996.8437079298141))
s.add_soft(c_21)
c_22 = Bool('c_22')
s.add(c_22 == And(-0.6841178532503372*x_1 + 0.3291598466918099*x_2 <= 1.6424972174628243, 1.0*x_1 <= 1.0, -1.0*x_1 <= 0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= 0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= 0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= 0.0, -1.0*x_3 <= 5.714999999999995e-07, 1.0*x_3 <= 2.8000000000000003e-05, -1.0*x_4 <= 2.2e-05, 1.0*x_4 <= 0.000978, -0.8421854431028581*x_1 + 0.286454183213154*x_2 <= 1008.7350771205056, 0.8421854431028581*x_1 + -0.286454183213154*x_2 <= 991.2649228794944, -0.5729085324850244*x_1 + -0.2965583040292779*x_2 <= 1003.1562920701859, 0.5729085324850244*x_1 + 0.2965583040292779*x_2 <= 996.8437079298141))
s.add_soft(c_22)
c_23 = Bool('c_23')
s.add(c_23 == And(-0.6702355239970705*x_1 + 0.3301950084948389*x_2 <= 1.5311471094581321, 1.0*x_1 <= 1.0, -1.0*x_1 <= 0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= 0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= 0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= 0.0, -1.0*x_3 <= 5.714999999999995e-07, 1.0*x_3 <= 2.8000000000000003e-05, -1.0*x_4 <= 2.2e-05, 1.0*x_4 <= 0.000978, -0.8421854431028581*x_1 + 0.286454183213154*x_2 <= 1008.7350771205056, 0.8421854431028581*x_1 + -0.286454183213154*x_2 <= 991.2649228794944, -0.5729085324850244*x_1 + -0.2965583040292779*x_2 <= 1003.1562920701859, 0.5729085324850244*x_1 + 0.2965583040292779*x_2 <= 996.8437079298141))
s.add_soft(c_23)
c_24 = Bool('c_24')
s.add(c_24 == And(-0.6563165485928734*x_1 + 0.3309025996219992*x_2 <= 1.417630579307982, 1.0*x_1 <= 1.0, -1.0*x_1 <= 0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= 0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= 0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= 0.0, -1.0*x_3 <= 5.714999999999995e-07, 1.0*x_3 <= 2.8000000000000003e-05, -1.0*x_4 <= 2.2e-05, 1.0*x_4 <= 0.000978, -0.8421854431028581*x_1 + 0.286454183213154*x_2 <= 1008.7350771205056, 0.8421854431028581*x_1 + -0.286454183213154*x_2 <= 991.2649228794944, -0.5729085324850244*x_1 + -0.2965583040292779*x_2 <= 1003.1562920701859, 0.5729085324850244*x_1 + 0.2965583040292779*x_2 <= 996.8437079298141))
s.add_soft(c_24)
c_25 = Bool('c_25')
s.add(c_25 == And(-0.6423744818441032*x_1 + 0.33129488296552584*x_2 <= 1.3021315980497565, 1.0*x_1 <= 1.0, -1.0*x_1 <= 0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= 0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= 0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= 0.0, -1.0*x_3 <= 5.714999999999995e-07, 1.0*x_3 <= 2.8000000000000003e-05, -1.0*x_4 <= 2.2e-05, 1.0*x_4 <= 0.000978, -0.8421854431028581*x_1 + 0.286454183213154*x_2 <= 1008.7350771205056, 0.8421854431028581*x_1 + -0.286454183213154*x_2 <= 991.2649228794944, -0.5729085324850244*x_1 + -0.2965583040292779*x_2 <= 1003.1562920701859, 0.5729085324850244*x_1 + 0.2965583040292779*x_2 <= 996.8437079298141))
s.add_soft(c_25)
c_26 = Bool('c_26')
s.add(c_26 == And(-0.6284223030406337*x_1 + 0.3313839076381118*x_2 <= 1.184828065311521, 1.0*x_1 <= 1.0, -1.0*x_1 <= 0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= 0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= 0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= 0.0, -1.0*x_3 <= 5.714999999999995e-07, 1.0*x_3 <= 2.8000000000000003e-05, -1.0*x_4 <= 2.2e-05, 1.0*x_4 <= 0.000978, -0.8421854431028581*x_1 + 0.286454183213154*x_2 <= 1008.7350771205056, 0.8421854431028581*x_1 + -0.286454183213154*x_2 <= 991.2649228794944, -0.5729085324850244*x_1 + -0.2965583040292779*x_2 <= 1003.1562920701859, 0.5729085324850244*x_1 + 0.2965583040292779*x_2 <= 996.8437079298141))
s.add_soft(c_26)
c_27 = Bool('c_27')
s.add(c_27 == And(-0.6144725517003691*x_1 + 0.33118151880255925*x_2 <= 1.0658930112240448, 1.0*x_1 <= 1.0, -1.0*x_1 <= 0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= 0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= 0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= 0.0, -1.0*x_3 <= 5.714999999999995e-07, 1.0*x_3 <= 2.8000000000000003e-05, -1.0*x_4 <= 2.2e-05, 1.0*x_4 <= 0.000978, -0.8421854431028581*x_1 + 0.286454183213154*x_2 <= 1008.7350771205056, 0.8421854431028581*x_1 + -0.286454183213154*x_2 <= 991.2649228794944, -0.5729085324850244*x_1 + -0.2965583040292779*x_2 <= 1003.1562920701859, 0.5729085324850244*x_1 + 0.2965583040292779*x_2 <= 996.8437079298141))
s.add_soft(c_27)
c_28 = Bool('c_28')
s.add(c_28 == And(-0.6005372524158638*x_1 + 0.33069934666802375*x_2 <= 0.9454938994670243, 1.0*x_1 <= 1.0, -1.0*x_1 <= 0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= 0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= 0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= 0.0, -1.0*x_3 <= 5.714999999999995e-07, 1.0*x_3 <= 2.8000000000000003e-05, -1.0*x_4 <= 2.2e-05, 1.0*x_4 <= 0.000978, -0.8421854431028581*x_1 + 0.286454183213154*x_2 <= 1008.7350771205056, 0.8421854431028581*x_1 + -0.286454183213154*x_2 <= 991.2649228794944, -0.5729085324850244*x_1 + -0.2965583040292779*x_2 <= 1003.1562920701859, 0.5729085324850244*x_1 + 0.2965583040292779*x_2 <= 996.8437079298141))
s.add_soft(c_28)
c_29 = Bool('c_29')
s.add(c_29 == And(-0.5866279442459148*x_1 + 0.32994880763466233*x_2 <= 0.8237928819322873, 1.0*x_1 <= 1.0, -1.0*x_1 <= 0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= 0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= 0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= 0.0, -1.0*x_3 <= 5.714999999999995e-07, 1.0*x_3 <= 2.8000000000000003e-05, -1.0*x_4 <= 2.2e-05, 1.0*x_4 <= 0.000978, -0.8421854431028581*x_1 + 0.286454183213154*x_2 <= 1008.7350771205056, 0.8421854431028581*x_1 + -0.286454183213154*x_2 <= 991.2649228794944, -0.5729085324850244*x_1 + -0.2965583040292779*x_2 <= 1003.1562920701859, 0.5729085324850244*x_1 + 0.2965583040292779*x_2 <= 996.8437079298141))
s.add_soft(c_29)
c_30 = Bool('c_30')
s.add(c_30 == And(-0.5727556910259557*x_1 + 0.328941102102057*x_2 <= 0.7009468733577968, 1.0*x_1 <= 1.0, -1.0*x_1 <= 0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= 0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= 0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= 0.0, -1.0*x_3 <= 5.714999999999995e-07, 1.0*x_3 <= 2.8000000000000003e-05, -1.0*x_4 <= 2.2e-05, 1.0*x_4 <= 0.000978, -0.8421854431028581*x_1 + 0.286454183213154*x_2 <= 1008.7350771205056, 0.8421854431028581*x_1 + -0.286454183213154*x_2 <= 991.2649228794944, -0.5729085324850244*x_1 + -0.2965583040292779*x_2 <= 1003.1562920701859, 0.5729085324850244*x_1 + 0.2965583040292779*x_2 <= 996.8437079298141))
s.add_soft(c_30)
c_31 = Bool('c_31')
s.add(c_31 == And(-0.5589310885053445*x_1 + 0.3276872142252823*x_2 <= 0.5771076102100317, 1.0*x_1 <= 1.0, -1.0*x_1 <= 0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= 0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= 0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= 0.0, -1.0*x_3 <= 5.714999999999995e-07, 1.0*x_3 <= 2.8000000000000003e-05, -1.0*x_4 <= 2.2e-05, 1.0*x_4 <= 0.000978, -0.8421854431028581*x_1 + 0.286454183213154*x_2 <= 1008.7350771205056, 0.8421854431028581*x_1 + -0.286454183213154*x_2 <= 991.2649228794944, -0.5729085324850244*x_1 + -0.2965583040292779*x_2 <= 1003.1562920701859, 0.5729085324850244*x_1 + 0.2965583040292779*x_2 <= 996.8437079298141))
s.add_soft(c_31)
c_32 = Bool('c_32')
s.add(c_32 == And(-0.5451642748455131*x_1 + 0.3261979099738224*x_2 <= 0.45242172832338756, 1.0*x_1 <= 1.0, -1.0*x_1 <= 0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= 0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= 0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= 0.0, -1.0*x_3 <= 5.714999999999995e-07, 1.0*x_3 <= 2.8000000000000003e-05, -1.0*x_4 <= 2.2e-05, 1.0*x_4 <= 0.000978, -0.8421854431028581*x_1 + 0.286454183213154*x_2 <= 1008.7350771205056, 0.8421854431028581*x_1 + -0.286454183213154*x_2 <= 991.2649228794944, -0.5729085324850244*x_1 + -0.2965583040292779*x_2 <= 1003.1562920701859, 0.5729085324850244*x_1 + 0.2965583040292779*x_2 <= 996.8437079298141))
s.add_soft(c_32)
c_33 = Bool('c_33')
s.add(c_33 == And(-0.5314649406675249*x_1 + 0.3244837362421323*x_2 <= 0.3270308426966757, 1.0*x_1 <= 1.0, -1.0*x_1 <= 0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= 0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= 0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= 0.0, -1.0*x_3 <= 5.714999999999995e-07, 1.0*x_3 <= 2.8000000000000003e-05, -1.0*x_4 <= 2.2e-05, 1.0*x_4 <= 0.000978, -0.8421854431028581*x_1 + 0.286454183213154*x_2 <= 1008.7350771205056, 0.8421854431028581*x_1 + -0.286454183213154*x_2 <= 991.2649228794944, -0.5729085324850244*x_1 + -0.2965583040292779*x_2 <= 1003.1562920701859, 0.5729085324850244*x_1 + 0.2965583040292779*x_2 <= 996.8437079298141))
s.add_soft(c_33)
c_34 = Bool('c_34')
s.add(c_34 == And(-0.5178423365004764*x_1 + 0.322555020870437*x_2 <= 0.2010716105032042, 1.0*x_1 <= 1.0, -1.0*x_1 <= 0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= 0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= 0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= 0.0, -1.0*x_3 <= 5.714999999999995e-07, 1.0*x_3 <= 2.8000000000000003e-05, -1.0*x_4 <= 2.2e-05, 1.0*x_4 <= 0.000978, -0.8421854431028581*x_1 + 0.286454183213154*x_2 <= 1008.7350771205056, 0.8421854431028581*x_1 + -0.286454183213154*x_2 <= 991.2649228794944, -0.5729085324850244*x_1 + -0.2965583040292779*x_2 <= 1003.1562920701859, 0.5729085324850244*x_1 + 0.2965583040292779*x_2 <= 996.8437079298141))
s.add_soft(c_34)
c_35 = Bool('c_35')
s.add(c_35 == And(-0.5043052787007054*x_1 + 0.32042187430729496*x_2 <= 0.07467579049593098, 1.0*x_1 <= 1.0, -1.0*x_1 <= 0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= 0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= 0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= 0.0, -1.0*x_3 <= 5.714999999999995e-07, 1.0*x_3 <= 2.8000000000000003e-05, -1.0*x_4 <= 2.2e-05, 1.0*x_4 <= 0.000978, -0.8421854431028581*x_1 + 0.286454183213154*x_2 <= 1008.7350771205056, 0.8421854431028581*x_1 + -0.286454183213154*x_2 <= 991.2649228794944, -0.5729085324850244*x_1 + -0.2965583040292779*x_2 <= 1003.1562920701859, 0.5729085324850244*x_1 + 0.2965583040292779*x_2 <= 996.8437079298141))
s.add_soft(c_35)
c_36 = Bool('c_36')
s.add(c_36 == And(-0.4908621633848351*x_1 + 0.31809418672893797*x_2 <= -0.052029655676314235, 1.0*x_1 <= 1.0, -1.0*x_1 <= 0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= 0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= 0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= 0.0, -1.0*x_3 <= 5.714999999999995e-07, 1.0*x_3 <= 2.8000000000000003e-05, -1.0*x_4 <= 2.2e-05, 1.0*x_4 <= 0.000978, -0.8421854431028581*x_1 + 0.286454183213154*x_2 <= 1008.7350771205056, 0.8421854431028581*x_1 + -0.286454183213154*x_2 <= 991.2649228794944, -0.5729085324850244*x_1 + -0.2965583040292779*x_2 <= 1003.1562920701859, 0.5729085324850244*x_1 + 0.2965583040292779*x_2 <= 996.8437079298141))
s.add_soft(c_36)
c_37 = Bool('c_37')
s.add(c_37 == And(-0.47752097870750077*x_1 + 0.31558162681362917*x_2 <= -0.17892246674734835, 1.0*x_1 <= 1.0, -1.0*x_1 <= 0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= 0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= 0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= 0.0, -1.0*x_3 <= 5.714999999999995e-07, 1.0*x_3 <= 2.8000000000000003e-05, -1.0*x_4 <= 2.2e-05, 1.0*x_4 <= 0.000978, -0.8421854431028581*x_1 + 0.286454183213154*x_2 <= 1008.7350771205056, 0.8421854431028581*x_1 + -0.286454183213154*x_2 <= 991.2649228794944, -0.5729085324850244*x_1 + -0.2965583040292779*x_2 <= 1003.1562920701859, 0.5729085324850244*x_1 + 0.2965583040292779*x_2 <= 996.8437079298141))
s.add_soft(c_37)
c_38 = Bool('c_38')
s.add(c_38 == And(-0.4642893144340876*x_1 + 0.3128936415879279*x_2 <= -0.3058850018756951, 1.0*x_1 <= 1.0, -1.0*x_1 <= 0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= 0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= 0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= 0.0, -1.0*x_3 <= 5.714999999999995e-07, 1.0*x_3 <= 2.8000000000000003e-05, -1.0*x_4 <= 2.2e-05, 1.0*x_4 <= 0.000978, -0.8421854431028581*x_1 + 0.286454183213154*x_2 <= 1008.7350771205056, 0.8421854431028581*x_1 + -0.286454183213154*x_2 <= 991.2649228794944, -0.5729085324850244*x_1 + -0.2965583040292779*x_2 <= 1003.1562920701859, 0.5729085324850244*x_1 + 0.2965583040292779*x_2 <= 996.8437079298141))
s.add_soft(c_38)
c_39 = Bool('c_39')
s.add(c_39 == And(-0.4511743762754283*x_1 + 0.310039467789345*x_2 <= -0.011813014654559417, 1.0*x_1 <= 1.0, -1.0*x_1 <= 0.0, 1.0*x_2 <= 1.0, -1.0*x_2 <= 0.0, 1.0*x_3 <= 0.0, -1.0*x_3 <= 0.0, 1.0*x_4 <= 0.0, -1.0*x_4 <= 0.0, -1.0*x_3 <= 5.714999999999995e-07, 1.0*x_3 <= 2.7999999999999996e-05, -1.0*x_4 <= 2.2e-05, 1.0*x_4 <= 0.000978, -0.8421854431028581*x_1 + 0.286454183213154*x_2 <= 1008.7350771205056, 0.8421854431028581*x_1 + -0.286454183213154*x_2 <= 991.2649228794944, -0.5729085324850244*x_1 + -0.2965583040292779*x_2 <= 1003.1562920701859, 0.5729085324850244*x_1 + 0.2965583040292779*x_2 <= 996.8437079298141, -1.0*x_3 <= 0.0, 1.0*x_3 <= 2.1428500000000013e-05, -1.0*x_4 <= 5e-05, 1.0*x_4 <= 0.00095, -0.4642893144340876*x_1 + 0.3128936415879279*x_2 <= 1005.6941149981243, 0.4642893144340876*x_1 + -0.3128936415879279*x_2 <= 994.3058850018757, -0.6257874524386793*x_1 + 0.13169868503491716*x_2 <= 1006.030568054196, 0.6257874524386793*x_1 + -0.13169868503491716*x_2 <= 993.969431945804))
s.add_soft(c_39)
if s.check() == sat:
	m = s.model()
	print(m)
else:
	print('No solution')