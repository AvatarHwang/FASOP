import numpy as np

data_1_1 = [0.0008275508880615234, 0.0010433197021484375, 0.0010268688201904297, 0.0010416507720947266, 0.0010509490966796875, 0.0010619163513183594, 0.001047372817993164, 0.001087188720703125, 0.0010416507720947266, 0.0009849071502685547, 0.0010535717010498047, 0.0010461807250976562, 0.0010523796081542969, 0.001062154769897461, 0.0010633468627929688, 0.0010428428649902344, 0.001073598861694336, 0.0010764598846435547, 0.001033782958984375, 0.001047372817993164, 0.0010504722595214844, 0.001043558120727539, 0.0010581016540527344, 0.0010356903076171875, 0.0010502338409423828, 0.001047372817993164, 0.0010495185852050781, 0.0010528564453125, 0.0010364055633544922, 0.0009930133819580078, 0.0009949207305908203, 0.0010712146759033203, 0.0010685920715332031, 0.0010542869567871094, 0.0010483264923095703, 0.000980377197265625, 0.0009999275207519531, 0.0010516643524169922, 0.0010554790496826172, 0.0010523796081542969, 0.0010428428649902344, 0.0010638236999511719, 0.0010714530944824219, 0.00104522705078125, 0.0010433197021484375, 0.0010569095611572266, 0.0010428428649902344, 0.0010428428649902344, 0.0008153915405273438, 9.870529174804688e-05]
data_1_2 = [0.0008275508880615234, 0.0010433197021484375, 0.0010268688201904297, 0.0010416507720947266, 0.0010509490966796875, 0.0010619163513183594, 0.001047372817993164, 0.001087188720703125, 0.0010416507720947266, 0.0009849071502685547, 0.0010535717010498047, 0.0010461807250976562, 0.0010523796081542969, 0.001062154769897461, 0.0010633468627929688, 0.0010428428649902344, 0.001073598861694336, 0.0010764598846435547, 0.001033782958984375, 0.001047372817993164, 0.0010504722595214844, 0.001043558120727539, 0.0010581016540527344, 0.0010356903076171875, 0.0010502338409423828, 0.001047372817993164, 0.0010495185852050781, 0.0010528564453125, 0.0010364055633544922, 0.0009930133819580078, 0.0009949207305908203, 0.0010712146759033203, 0.0010685920715332031, 0.0010542869567871094, 0.0010483264923095703, 0.000980377197265625, 0.0009999275207519531, 0.0010516643524169922, 0.0010554790496826172, 0.0010523796081542969, 0.0010428428649902344, 0.0010638236999511719, 0.0010714530944824219, 0.00104522705078125, 0.0010433197021484375, 0.0010569095611572266, 0.0010428428649902344, 0.0010428428649902344, 0.0008153915405273438, 9.870529174804688e-05]
data_1_3 = [0.0008275508880615234, 0.0010433197021484375, 0.0010268688201904297, 0.0010416507720947266, 0.0010509490966796875, 0.0010619163513183594, 0.001047372817993164, 0.001087188720703125, 0.0010416507720947266, 0.0009849071502685547, 0.0010535717010498047, 0.0010461807250976562, 0.0010523796081542969, 0.001062154769897461, 0.0010633468627929688, 0.0010428428649902344, 0.001073598861694336, 0.0010764598846435547, 0.001033782958984375, 0.001047372817993164, 0.0010504722595214844, 0.001043558120727539, 0.0010581016540527344, 0.0010356903076171875, 0.0010502338409423828, 0.001047372817993164, 0.0010495185852050781, 0.0010528564453125, 0.0010364055633544922, 0.0009930133819580078, 0.0009949207305908203, 0.0010712146759033203, 0.0010685920715332031, 0.0010542869567871094, 0.0010483264923095703, 0.000980377197265625, 0.0009999275207519531, 0.0010516643524169922, 0.0010554790496826172, 0.0010523796081542969, 0.0010428428649902344, 0.0010638236999511719, 0.0010714530944824219, 0.00104522705078125, 0.0010433197021484375, 0.0010569095611572266, 0.0010428428649902344, 0.0010428428649902344, 0.0008153915405273438, 9.870529174804688e-05]

data_1 = [data_1_1[i] + data_1_2[i] + data_1_3[i] for i in range(len(data_1_1))]
data_1 = [i/3 for i in data_1]

np.save('/home/soonyear/AMP/src/known_cost/gpt2XL_A100_1.npy', data_1)

data_2_1 =[0.0004227161407470703, 0.0006957054138183594, 0.0005373954772949219, 0.0005292892456054688, 0.0005154609680175781, 0.0006935596466064453, 0.0018105506896972656, 0.0007481575012207031, 0.0005354881286621094, 0.0006411075592041016, 0.0005393028259277344, 0.0007717609405517578, 0.0007314682006835938, 0.0007925033569335938, 0.0005886554718017578, 0.0005407333374023438, 0.0031690597534179688, 0.0007512569427490234, 0.0007078647613525391, 0.0007915496826171875, 0.00052642822265625, 0.0006322860717773438, 0.0005505084991455078, 0.0007004737854003906, 0.0007050037384033203, 0.0008106231689453125, 0.0005435943603515625, 0.0005340576171875, 0.0006139278411865234, 0.0008220672607421875, 0.0006949901580810547, 0.0007379055023193359, 0.0007061958312988281, 0.0007252693176269531, 0.00054931640625, 0.0006723403930664062, 0.0007195472717285156, 0.0007212162017822266, 0.0006296634674072266, 0.0005490779876708984, 0.0005333423614501953, 0.0008499622344970703, 0.0006961822509765625, 0.0007030963897705078, 0.0006380081176757812, 0.0007519721984863281, 0.0006198883056640625, 0.0007171630859375, 0.0003943443298339844, 0.00010633468627929688]
data_2_2 =[0.0004227161407470703, 0.0006957054138183594, 0.0005373954772949219, 0.0005292892456054688, 0.0005154609680175781, 0.0006935596466064453, 0.0018105506896972656, 0.0007481575012207031, 0.0005354881286621094, 0.0006411075592041016, 0.0005393028259277344, 0.0007717609405517578, 0.0007314682006835938, 0.0007925033569335938, 0.0005886554718017578, 0.0005407333374023438, 0.0031690597534179688, 0.0007512569427490234, 0.0007078647613525391, 0.0007915496826171875, 0.00052642822265625, 0.0006322860717773438, 0.0005505084991455078, 0.0007004737854003906, 0.0007050037384033203, 0.0008106231689453125, 0.0005435943603515625, 0.0005340576171875, 0.0006139278411865234, 0.0008220672607421875, 0.0006949901580810547, 0.0007379055023193359, 0.0007061958312988281, 0.0007252693176269531, 0.00054931640625, 0.0006723403930664062, 0.0007195472717285156, 0.0007212162017822266, 0.0006296634674072266, 0.0005490779876708984, 0.0005333423614501953, 0.0008499622344970703, 0.0006961822509765625, 0.0007030963897705078, 0.0006380081176757812, 0.0007519721984863281, 0.0006198883056640625, 0.0007171630859375, 0.0003943443298339844, 0.00010633468627929688]
data_2_3 =[0.0004227161407470703, 0.0006957054138183594, 0.0005373954772949219, 0.0005292892456054688, 0.0005154609680175781, 0.0006935596466064453, 0.0018105506896972656, 0.0007481575012207031, 0.0005354881286621094, 0.0006411075592041016, 0.0005393028259277344, 0.0007717609405517578, 0.0007314682006835938, 0.0007925033569335938, 0.0005886554718017578, 0.0005407333374023438, 0.0031690597534179688, 0.0007512569427490234, 0.0007078647613525391, 0.0007915496826171875, 0.00052642822265625, 0.0006322860717773438, 0.0005505084991455078, 0.0007004737854003906, 0.0007050037384033203, 0.0008106231689453125, 0.0005435943603515625, 0.0005340576171875, 0.0006139278411865234, 0.0008220672607421875, 0.0006949901580810547, 0.0007379055023193359, 0.0007061958312988281, 0.0007252693176269531, 0.00054931640625, 0.0006723403930664062, 0.0007195472717285156, 0.0007212162017822266, 0.0006296634674072266, 0.0005490779876708984, 0.0005333423614501953, 0.0008499622344970703, 0.0006961822509765625, 0.0007030963897705078, 0.0006380081176757812, 0.0007519721984863281, 0.0006198883056640625, 0.0007171630859375, 0.0003943443298339844, 0.00010633468627929688]

#elementwise addition
data_2 = [data_2_1[i] + data_2_2[i] + data_2_3[i] for i in range(len(data_2_1))]
data_2 = [i/3 for i in data_2]

np.save('/home/soonyear/AMP/src/known_cost/gpt2XL_A100_2.npy', data_2)

data_4_1 =[0.0441741943359375, 0.0003955364227294922, 0.00036644935607910156, 0.00040531158447265625, 0.0003745555877685547, 0.0004181861877441406, 0.002239704132080078, 0.0005164146423339844, 0.0003647804260253906, 0.00045037269592285156, 0.00037384033203125, 0.00043845176696777344, 0.0004074573516845703, 0.0006954669952392578, 0.0003654956817626953, 0.0004246234893798828, 0.0003590583801269531, 0.00043129920959472656, 0.0004248619079589844, 0.00046324729919433594, 0.0003590583801269531, 0.00041961669921875, 0.0003769397735595703, 0.0004286766052246094, 0.00034236907958984375, 0.0004665851593017578, 0.00035834312438964844, 0.00042438507080078125, 0.00035858154296875, 0.0004215240478515625, 0.0004639625549316406, 0.0003986358642578125, 0.0006852149963378906, 0.0004582405090332031, 0.0006225109100341797, 0.0003952980041503906, 0.00037407875061035156, 0.0004303455352783203, 0.000362396240234375, 0.0004591941833496094, 0.00035953521728515625, 0.00043320655822753906, 0.0003533363342285156, 0.00041294097900390625, 0.0004296302795410156, 0.0005428791046142578, 0.0004131793975830078, 0.00043201446533203125, 0.00030517578125, 0.0033473968505859375]
data_4_2 =[0.0441741943359375, 0.0003955364227294922, 0.00036644935607910156, 0.00040531158447265625, 0.0003745555877685547, 0.0004181861877441406, 0.002239704132080078, 0.0005164146423339844, 0.0003647804260253906, 0.00045037269592285156, 0.00037384033203125, 0.00043845176696777344, 0.0004074573516845703, 0.0006954669952392578, 0.0003654956817626953, 0.0004246234893798828, 0.0003590583801269531, 0.00043129920959472656, 0.0004248619079589844, 0.00046324729919433594, 0.0003590583801269531, 0.00041961669921875, 0.0003769397735595703, 0.0004286766052246094, 0.00034236907958984375, 0.0004665851593017578, 0.00035834312438964844, 0.00042438507080078125, 0.00035858154296875, 0.0004215240478515625, 0.0004639625549316406, 0.0003986358642578125, 0.0006852149963378906, 0.0004582405090332031, 0.0006225109100341797, 0.0003952980041503906, 0.00037407875061035156, 0.0004303455352783203, 0.000362396240234375, 0.0004591941833496094, 0.00035953521728515625, 0.00043320655822753906, 0.0003533363342285156, 0.00041294097900390625, 0.0004296302795410156, 0.0005428791046142578, 0.0004131793975830078, 0.00043201446533203125, 0.00030517578125, 0.0033473968505859375]
data_4_3 =[0.0441741943359375, 0.0003955364227294922, 0.00036644935607910156, 0.00040531158447265625, 0.0003745555877685547, 0.0004181861877441406, 0.002239704132080078, 0.0005164146423339844, 0.0003647804260253906, 0.00045037269592285156, 0.00037384033203125, 0.00043845176696777344, 0.0004074573516845703, 0.0006954669952392578, 0.0003654956817626953, 0.0004246234893798828, 0.0003590583801269531, 0.00043129920959472656, 0.0004248619079589844, 0.00046324729919433594, 0.0003590583801269531, 0.00041961669921875, 0.0003769397735595703, 0.0004286766052246094, 0.00034236907958984375, 0.0004665851593017578, 0.00035834312438964844, 0.00042438507080078125, 0.00035858154296875, 0.0004215240478515625, 0.0004639625549316406, 0.0003986358642578125, 0.0006852149963378906, 0.0004582405090332031, 0.0006225109100341797, 0.0003952980041503906, 0.00037407875061035156, 0.0004303455352783203, 0.000362396240234375, 0.0004591941833496094, 0.00035953521728515625, 0.00043320655822753906, 0.0003533363342285156, 0.00041294097900390625, 0.0004296302795410156, 0.0005428791046142578, 0.0004131793975830078, 0.00043201446533203125, 0.00030517578125, 0.0033473968505859375]

data_4 = [data_4_1[i]+data_4_2[i]+data_4_3[i] for i in range(len(data_4_1))]
data_4 = [i/3 for i in data_4]

np.save('/home/soonyear/AMP/src/known_cost/gpt2XL_A100_4.npy', data_4)