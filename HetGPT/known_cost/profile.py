import numpy as np

data_1_1 = [1.2174392123692235, 0.004522806134731749, 0.00430398774719463, 0.004151045652741553, 0.004161535508328184, 0.004098176780584934, 0.004216292554490396, 0.004114121361076612, 0.0041445419422778415, 0.004037125821070742, 0.004044678517093116, 0.004066287619601576, 0.0040404825748584635, 0.004019712660796935, 0.0040268457625958435, 0.004022649820361192, 0.004046566691098709, 0.004090414287450827, 0.0039779630355621445, 0.00422300606206584, 0.004082861591428452, 0.003987194108478379, 0.00427188878909954, 0.004039643386411534, 0.004058105532244003, 0.004096917997914538, 0.003979431615344272, 0.004137199043367201, 0.004090414287450827, 0.004090833881674292, 0.004080344026087661, 0.004028524139489705, 0.004025167385701983, 0.004076148083853009, 0.003986984311366647, 0.004048454865104303, 0.004071952141618356, 0.003986145122919716, 0.00408265179431672, 0.004068805184942367, 0.003989501876707438, 0.004097966983473201, 0.004018244081014807, 0.004094190635462014, 0.004109715621730227, 0.003985515731584518, 0.004063770054260785, 0.004036916023959009, 0.02023619020928127, 0.00025930923010151417]
data_1_2 = [1.2174392123692235, 0.004522806134731749, 0.00430398774719463, 0.004151045652741553, 0.004161535508328184, 0.004098176780584934, 0.004216292554490396, 0.004114121361076612, 0.0041445419422778415, 0.004037125821070742, 0.004044678517093116, 0.004066287619601576, 0.0040404825748584635, 0.004019712660796935, 0.0040268457625958435, 0.004022649820361192, 0.004046566691098709, 0.004090414287450827, 0.0039779630355621445, 0.00422300606206584, 0.004082861591428452, 0.003987194108478379, 0.00427188878909954, 0.004039643386411534, 0.004058105532244003, 0.004096917997914538, 0.003979431615344272, 0.004137199043367201, 0.004090414287450827, 0.004090833881674292, 0.004080344026087661, 0.004028524139489705, 0.004025167385701983, 0.004076148083853009, 0.003986984311366647, 0.004048454865104303, 0.004071952141618356, 0.003986145122919716, 0.00408265179431672, 0.004068805184942367, 0.003989501876707438, 0.004097966983473201, 0.004018244081014807, 0.004094190635462014, 0.004109715621730227, 0.003985515731584518, 0.004063770054260785, 0.004036916023959009, 0.02023619020928127, 0.00025930923010151417]
data_1_3 = [1.2174392123692235, 0.004522806134731749, 0.00430398774719463, 0.004151045652741553, 0.004161535508328184, 0.004098176780584934, 0.004216292554490396, 0.004114121361076612, 0.0041445419422778415, 0.004037125821070742, 0.004044678517093116, 0.004066287619601576, 0.0040404825748584635, 0.004019712660796935, 0.0040268457625958435, 0.004022649820361192, 0.004046566691098709, 0.004090414287450827, 0.0039779630355621445, 0.00422300606206584, 0.004082861591428452, 0.003987194108478379, 0.00427188878909954, 0.004039643386411534, 0.004058105532244003, 0.004096917997914538, 0.003979431615344272, 0.004137199043367201, 0.004090414287450827, 0.004090833881674292, 0.004080344026087661, 0.004028524139489705, 0.004025167385701983, 0.004076148083853009, 0.003986984311366647, 0.004048454865104303, 0.004071952141618356, 0.003986145122919716, 0.00408265179431672, 0.004068805184942367, 0.003989501876707438, 0.004097966983473201, 0.004018244081014807, 0.004094190635462014, 0.004109715621730227, 0.003985515731584518, 0.004063770054260785, 0.004036916023959009, 0.02023619020928127, 0.00025930923010151417]

data_1 = [data_1_1[i] + data_1_2[i] + data_1_3[i] for i in range(len(data_1_1))]
data_1 = [i/3 for i in data_1]

np.save('/home/soonyear/AMP/src/known_cost/gpt2XL_A10_1.npy', data_1)

data_2_1 =[0.5701198990301796, 0.002292172323538012, 0.0016923880066471004, 0.0016976851411470667, 0.0016950830399891886, 0.0017911749184622598, 0.0018210990817778584, 0.0018382915358566962, 0.0015828209614635888, 0.0017216616446732295, 0.001688856583647123, 0.0018534394818829154, 0.0018224930645410074, 0.00232311874087992, 0.0023973715560636566, 0.0016924809388313105, 0.0016819796020155879, 0.0018293700461725427, 0.0018089249656463573, 0.0018003752046990432, 0.001698707395173376, 0.0016766824675156218, 0.0015955526707003497, 0.001932431838461359, 0.0018815979336985252, 0.0018354106381461884, 0.0017109744434890871, 0.0017305831343573832, 0.001716736238910103, 0.001868029834803875, 0.0018440533312777119, 0.0018181252518831405, 0.001676217806594572, 0.001727981033199505, 0.0016909940238839514, 0.0017897809356991108, 0.002077870706749905, 0.0018927497958037174, 0.0016934102606734097, 0.001711253240041717, 0.0017006589710417846, 0.0018842929670406133, 0.0018388491289619557, 0.0018145008966989533, 0.0017108815113048774, 0.0016978710055154866, 0.0016912728204365813, 0.0018291841818041225, 0.006083247846198042, 0.0003522129781556483]
data_2_2 =[0.5706826034055706, 0.0021310279161179868, 0.0017809523781991672, 0.0017779785483044493, 0.001798144832278005, 0.0018479564830145294, 0.0018898688980932092, 0.0018847576279616628, 0.0016779835180945606, 0.0018146867610673732, 0.0017930335621464586, 0.0018288124530672829, 0.001866635852040726, 0.0023763688824322113, 0.0017351368113836699, 0.0017507494183309387, 0.001740433945883636, 0.0018791816969090669, 0.0018741633589617307, 0.0018393137898830055, 0.0017429431148573045, 0.0017300255412521235, 0.001669247892778827, 0.0019216517050930065, 0.001824723436962046, 0.0019035299291720695, 0.0017260294573310964, 0.0017137624090153851, 0.001779744259804438, 0.0018395925864356354, 0.0018920063383300377, 0.0018696096819354438, 0.0017294679481468641, 0.0017974013748043253, 0.0017452664194625527, 0.0018547405324618542, 0.0021063079551181447, 0.0019566871385401515, 0.0017609719585940315, 0.0017810453103833772, 0.0017274234400942454, 0.001947765648855998, 0.0019133807406983225, 0.0018745350876985703, 0.001770079312646605, 0.001768313601146616, 0.00174889077464674, 0.0019101281142509749, 0.006122744024487264, 0.0003827876667607164]
data_2_3 =[0.5701198990301796, 0.002292172323538012, 0.0016923880066471004, 0.0016976851411470667, 0.0016950830399891886, 0.0017911749184622598, 0.0018210990817778584, 0.0018382915358566962, 0.0015828209614635888, 0.0017216616446732295, 0.001688856583647123, 0.0018534394818829154, 0.0018224930645410074, 0.00232311874087992, 0.0023973715560636566, 0.0016924809388313105, 0.0016819796020155879, 0.0018293700461725427, 0.0018089249656463573, 0.0018003752046990432, 0.001698707395173376, 0.0016766824675156218, 0.0015955526707003497, 0.001932431838461359, 0.0018815979336985252, 0.0018354106381461884, 0.0017109744434890871, 0.0017305831343573832, 0.001716736238910103, 0.001868029834803875, 0.0018440533312777119, 0.0018181252518831405, 0.001676217806594572, 0.001727981033199505, 0.0016909940238839514, 0.0017897809356991108, 0.002077870706749905, 0.0018927497958037174, 0.0016934102606734097, 0.001711253240041717, 0.0017006589710417846, 0.0018842929670406133, 0.0018388491289619557, 0.0018145008966989533, 0.0017108815113048774, 0.0016978710055154866, 0.0016912728204365813, 0.0018291841818041225, 0.006083247846198042, 0.0003522129781556483]

#elementwise addition
data_2 = [data_2_1[i] + data_2_2[i] + data_2_3[i] for i in range(len(data_2_1))]
data_2 = [i/3 for i in data_2]

np.save('/home/soonyear/AMP/src/known_cost/gpt2XL_A10_2.npy', data_2)

data_4_1 =[0.29254633145034625, 0.007345460793181089, 0.0014544932085564716, 0.0010207991735297671, 0.0018784640046304488, 0.0019229466485349532, 0.0013794316827281737, 0.0008155777678077187, 0.00139690532954177, 0.0012000919420445434, 0.0010999003058794875, 0.0008644288234156222, 0.0015103900894924381, 0.0013936172777220073, 0.001060772489224311, 0.0014224581893982117, 0.0013421827528271473, 0.0009000806995756211, 0.0012862388997223271, 0.0015243877958108563, 0.0015617306700495904, 0.0010126729883180678, 0.0014525673496334677, 0.0013488058286355265, 0.0011700767261469951, 0.001005016524794906, 0.001564267167167693, 0.0011172800083553763, 0.0012067150178529226, 0.0011883488998311051, 0.0013800892930921262, 0.000933806716812616, 0.0011205210880062854, 0.001327762296989045, 0.0013906110589153669, 0.0008752793944208393, 0.0015497997391607369, 0.001235368040853712, 0.0011389811503658104, 0.001055088856793007, 0.0017250998733229442, 0.0013310503488088079, 0.0011677281177043073, 0.0013137176185017726, 0.001420250497462085, 0.0010190612032821781, 0.0010671137320195678, 0.001337062786422088, 0.003123555284436888, 0.00023692761969833172]
data_4_2 =[0.2923279108651763, 0.001446038218162796, 0.0014672696384846922, 0.0009967024509077917, 0.0023726581931407874, 0.0018939178481833339, 0.0013621459245899924, 0.0008238448695259794, 0.0013844107326266715, 0.001151475747280909, 0.0011185482569144277, 0.0008889482955572815, 0.0014107151471847734, 0.001296713693376714, 0.0013733722729460393, 0.0014156941970832713, 0.0012814477384992442, 0.0008911090153245542, 0.0012206187798336336, 0.0014828174263752847, 0.0015290380405273778, 0.0010232417263101622, 0.001442374388992203, 0.001307752153057346, 0.0011698888374715802, 0.0010169004835149055, 0.0015228846864075362, 0.0010997593893729263, 0.0011489862223316599, 0.0012092984871398791, 0.0013844107326266715, 0.0009153466544530909, 0.0010734549748148245, 0.001285299456345252, 0.0013245681895069899, 0.000907690190929929, 0.0014605995905074595, 0.0012488490533147396, 0.0011010276379319777, 0.0010293011360922963, 0.001693722464528637, 0.0013037125465359233, 0.0011531197731907902, 0.0013445783334386887, 0.0014108560636913344, 0.0010115926284344314, 0.0010783870525444687, 0.0013283729351841436, 0.003083488024404637, 0.0002287074901489249]
data_4_3 =[0.29252153014519144, 0.0036471540506496764, 0.0015697159387547284, 0.001014363986396803, 0.0018305523923996204, 0.0019219602329890245, 0.0013378143411237483, 0.0009053415824872414, 0.0014021662124533902, 0.0012045073259167962, 0.0011256410544113445, 0.0008908741544802854, 0.0014329799552214523, 0.0014337784820919663, 0.000997360061271744, 0.0014377711164445352, 0.0012644907855430393, 0.0009332430507863708, 0.0013025852144834332, 0.0015490481844590768, 0.0015557652046051635, 0.0010142230698902417, 0.0014405894465757605, 0.0014035753775190028, 0.0011404842597691305, 0.0010165716783329293, 0.0015691053005596296, 0.0011946431704575081, 0.0011860472635572712, 0.0012076544612299976, 0.0014061588468059594, 0.000957574634252615, 0.001085526822210239, 0.0013598912604850123, 0.0014313829014804247, 0.0008847677725292974, 0.001560838198841369, 0.0012628937318020117, 0.0011694660879518963, 0.0010662212608113465, 0.001711665833030771, 0.0013218438037134721, 0.0011657552866124498, 0.001297700108922643, 0.0015096385347907778, 0.0010605846005488963, 0.0010829433529232827, 0.0016092195327607349, 0.003173251839084159, 0.00022762713026528857]

data_4 = [data_4_1[i]+data_4_2[i]+data_4_3[i] for i in range(len(data_4_1))]
data_4 = [i/3 for i in data_4]

np.save('/home/soonyear/AMP/src/known_cost/gpt2XL_A10_4.npy', data_4)

print(len(data_4))