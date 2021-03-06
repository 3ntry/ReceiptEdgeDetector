import re

RAW_DATA = """Line [[370, 318],[370, 289]] -inf
Line [[370, 317],[370, 288]] -inf
Line [[65, 201],[94, 201]] 0
Line [[143, 201],[170, 201]] 0
Line [[143, 200],[172, 200]] 0
Line [[369, 361],[369, 337]] -inf
Line [[370, 354],[370, 332]] -inf
Line [[369, 366],[370, 337]] -29
Line [[366, 509],[367, 483]] -26
Line [[366, 488],[367, 509]] 21
Line [[366, 529],[366, 500]] -inf
Line [[367, 529],[367, 500]] -inf
Line [[145, 201],[170, 201]] 0
Line [[129, 201],[158, 200]] -0.0344828
Line [[255, 199],[284, 199]] 0
Line [[261, 198],[284, 198]] 0
Line [[372, 222],[373, 195]] -27
Line [[345, 196],[370, 195]] -0.04
Line [[85, 201],[114, 201]] 0
Line [[67, 529],[67, 500]] -inf
Line [[370, 340],[370, 311]] -inf
Line [[370, 341],[370, 312]] -inf
Line [[321, 196],[350, 197]] 0.0344828
Line [[175, 200],[204, 200]] 0
Line [[215, 200],[242, 199]] -0.037037
Line [[368, 488],[368, 460]] -inf
Line [[367, 489],[367, 464]] -inf
Line [[105, 201],[134, 201]] 0
Line [[370, 308],[372, 279]] -14.5
Line [[370, 279],[371, 304]] 25
Line [[370, 304],[372, 275]] -14.5
Line [[370, 303],[371, 275]] -28
Line [[279, 197],[300, 197]] 0
Line [[65, 361],[65, 332]] -inf
Line [[64, 361],[64, 332]] -inf
Line [[199, 200],[228, 200]] 0
Line [[208, 199],[228, 199]] 0
Line [[64, 309],[65, 329]] 20
Line [[63, 329],[63, 300]] -inf
Line [[208, 199],[236, 200]] 0.0357143
Line [[212, 199],[236, 199]] 0
Line [[66, 457],[66, 428]] -inf
Line [[65, 444],[67, 473]] 14.5
Line [[61, 220],[62, 249]] 29
Line [[367, 529],[367, 500]] -inf
Line [[366, 529],[366, 500]] -inf
Line [[365, 549],[366, 520]] -29
Line [[366, 549],[367, 520]] -29
Line [[365, 568],[366, 540]] -28
Line [[370, 329],[370, 300]] -inf
Line [[370, 329],[370, 300]] -inf
Line [[66, 471],[66, 442]] -inf
Line [[66, 489],[67, 460]] -29
Line [[65, 460],[67, 489]] 14.5
Line [[62, 268],[63, 293]] 25
Line [[66, 443],[66, 414]] -inf
Line [[65, 457],[66, 428]] -29
Line [[65, 436],[66, 457]] 21
Line [[67, 529],[67, 500]] -inf
Line [[67, 520],[67, 548]] inf
Line [[329, 196],[358, 195]] -0.0344828
Line [[329, 197],[350, 197]] 0
Line [[68, 589],[68, 560]] -inf
Line [[369, 372],[369, 343]] -inf
Line [[369, 379],[369, 350]] -inf
Line [[368, 469],[368, 440]] -inf
Line [[365, 589],[365, 560]] -inf
Line [[366, 584],[366, 560]] -inf
Line [[287, 198],[316, 197]] -0.0344828
Line [[287, 197],[316, 198]] 0.0344828
Line [[131, 201],[160, 201]] 0
Line [[113, 201],[142, 200]] -0.0344828
Line [[150, 201],[170, 201]] 0
Line [[135, 201],[164, 201]] 0
Line [[370, 350],[370, 321]] -inf
Line [[370, 354],[370, 325]] -inf
Line [[304, 198],[332, 197]] -0.0357143
Line [[303, 197],[332, 196]] -0.0344828
Line [[311, 198],[340, 197]] -0.0344828
Line [[311, 197],[338, 196]] -0.037037
Line [[371, 265],[372, 236]] -29
Line [[371, 241],[372, 265]] 24
Line [[373, 257],[373, 236]] -inf
Line [[371, 254],[372, 225]] -29
Line [[373, 254],[373, 225]] -inf
Line [[239, 199],[268, 199]] 0
Line [[64, 377],[65, 348]] -29
Line [[64, 356],[65, 377]] 21
Line [[263, 199],[289, 197]] -0.0769231
Line [[68, 569],[68, 540]] -inf
Line [[68, 589],[68, 560]] -inf
Line [[368, 488],[368, 460]] -inf
Line [[367, 489],[367, 464]] -inf
Line [[62, 265],[62, 236]] -inf
Line [[62, 252],[63, 281]] 29
Line [[85, 201],[114, 201]] 0
Line [[65, 201],[94, 201]] 0
Line [[167, 200],[196, 200]] 0
Line [[271, 198],[300, 197]] -0.0344828
Line [[159, 201],[188, 200]] -0.0344828
Line [[368, 425],[368, 396]] -inf
Line [[368, 441],[368, 412]] -inf
Line [[247, 199],[276, 199]] 0
Line [[63, 313],[63, 284]] -inf
Line [[371, 254],[372, 225]] -29
Line [[373, 254],[373, 225]] -inf
Line [[372, 213],[373, 242]] 29
Line [[372, 285],[372, 257]] -inf
Line [[371, 286],[371, 257]] -inf
Line [[371, 250],[372, 279]] 29
Line [[371, 274],[372, 250]] -24
Line [[151, 201],[180, 200]] -0.0344828
Line [[66, 414],[66, 385]] -inf
Line [[65, 396],[66, 425]] 29
Line [[65, 425],[66, 396]] -29
Line [[369, 393],[369, 364]] -inf
Line [[369, 404],[369, 375]] -inf
Line [[138, 201],[167, 201]] 0
Line [[121, 201],[150, 201]] 0
Line [[372, 243],[372, 214]] -inf
Line [[373, 243],[373, 214]] -inf
Line [[372, 208],[373, 229]] 21
Line [[223, 199],[252, 199]] 0
Line [[191, 200],[220, 199]] -0.0344828
Line [[370, 297],[371, 274]] -23
Line [[370, 288],[372, 263]] -12.5
Line [[369, 383],[369, 354]] -inf
Line [[369, 391],[369, 362]] -inf
Line [[368, 447],[368, 418]] -inf
Line [[368, 466],[368, 437]] -inf
Line [[366, 549],[367, 520]] -29
Line [[365, 549],[365, 523]] -inf
Line [[365, 569],[366, 540]] -29
Line [[365, 548],[366, 569]] 21
Line [[365, 589],[365, 560]] -inf
Line [[366, 584],[366, 560]] -inf
Line [[64, 393],[64, 364]] -inf
Line [[65, 364],[66, 393]] 29
Line [[231, 199],[260, 199]] 0
Line [[372, 233],[373, 204]] -29
Line [[373, 217],[374, 196]] -21
Line [[372, 195],[374, 217]] 11
Line [[372, 217],[373, 195]] -22
Line [[371, 275],[371, 246]] -inf
Line [[372, 275],[372, 246]] -inf
Line [[372, 267],[372, 238]] -inf
Line [[371, 267],[371, 241]] -inf
Line [[66, 400],[66, 378]] -inf
Line [[65, 409],[65, 380]] -inf
Line [[66, 409],[66, 380]] -inf
Line [[64, 405],[64, 380]] -inf
Line [[143, 201],[170, 201]] 0
Line [[143, 200],[172, 200]] 0
Line [[315, 197],[344, 197]] 0
Line [[315, 196],[344, 196]] 0
Line [[368, 389],[369, 410]] 21
Line [[368, 415],[368, 392]] -inf
Line [[368, 429],[368, 400]] -inf
Line [[295, 198],[319, 196]] -0.0833333
Line [[297, 198],[322, 198]] 0
Line [[183, 200],[212, 200]] 0
Line [[66, 429],[66, 400]] -inf
Line [[65, 441],[66, 412]] -29
Line [[65, 420],[66, 441]] 21
Line [[66, 456],[67, 485]] 29
Line [[67, 484],[67, 460]] -inf
Line [[66, 476],[67, 505]] 29
Line [[368, 488],[368, 460]] -inf
Line [[367, 489],[367, 464]] -inf
Line [[68, 569],[68, 540]] -inf
Line [[68, 589],[68, 560]] -inf
Line [[66, 480],[67, 509]] 29
Line [[67, 529],[67, 500]] -inf
Line [[337, 197],[362, 195]] -0.08
Line [[368, 469],[368, 440]] -inf
Line [[368, 488],[368, 460]] -inf
Line [[367, 489],[367, 464]] -inf
Line [[67, 520],[67, 548]] inf
Line [[68, 569],[68, 540]] -inf
Line [[368, 436],[368, 407]] -inf
Line [[368, 454],[368, 425]] -inf
Line [[63, 316],[64, 345]] 29
Line [[65, 345],[65, 320]] -inf
Line [[368, 404],[369, 375]] -29
Line [[368, 416],[368, 389]] -inf
Line [[369, 410],[369, 387]] -inf
Line [[105, 201],[134, 201]] 0
Line [[85, 201],[114, 201]] 0"""

lines = list()

dataLines = RAW_DATA.split('\n');

for data in dataLines:
    print data
    #matches = re.match("Line \[\[(\d),(\d)\],\[(\d),(\d)\]\] (\w)", data)
    # lines.append(
    #     (
    #         (
    #             (matches.group(1), matches.group(2)),
    #             (matches.group(3), matches.group(4))
    #         ),
    #             matches.group(5)
    #     )
    # )

for line in lines:
    print line
