# HW3


# Character-Level Language Modeling with RNN and LSTM
## - Shakespeare Dataset을 활용한 실험 -

### 1. 소개
이 보고서에서는 Shakespeare 데이터셋을 사용하여 문자 단위 언어 모델링 실험을 수행한 결과를 다룹니다. 바닐라 RNN(Recurrent Neural Network)과 LSTM(Long Short-Term Memory) 모델을 구현하고, 이들의 성능을 비교 분석하였습니다.

### 2. 데이터셋
실험에는 Shakespeare 데이터셋을 사용하였습니다. 이 데이터셋은 Shakespeare의 작품에서 발췌한 텍스트로 구성되어 있습니다. 데이터셋은 문자 단위로 처리되었으며, 각 문자는 고유한 인덱스로 매핑되었습니다.

### 3. 모델 구조
바닐라 RNN (CharRNN): 기본적인 RNN 구조로, 임베딩 층, RNN 층, 출력 층으로 구성됩니다.

- hidden layer size: 512
- number of layers: 3
  
LSTM (CharLSTM): RNN 대신 LSTM 층을 사용한 모델로, 장기 의존성을 더 잘 포착할 수 있습니다.

- hidden layer size: 512
- number of layers: 3

Optimization 초기에 모델은 hidden layer size = 128, number of layers = 5로 Adam 사용하여 학습을 진행했습니다. LSTM이 RNN에 비해 성능이 떨어지는 현상이 관찰되었습니다. 이를 해결하기위해 아래와 같이 설정하였습니다.

- hidden layer size를 128 -> 512
- number of layer를 5 -> 3
- Adam -> AdamW 

##### Hidden Layer Size 증가:
Hidden layer size를 증가시키면 모델의 표현력이 커집니다. 더 많은 뉴런을 사용하여 더 복잡한 패턴을 학습할 수 있게 됩니다.
특히 Shakespeare 데이터셋과 같이 어휘량이 많고 문맥 의존성이 높은 데이터에서는 더 큰 hidden layer size가 도움이 될 수 있습니다.

##### Number of Layers 감소:
초기에 5개의 층을 사용했을 때 LSTM의 성능이 RNN보다 낮았던 것은, LSTM의 깊이가 너무 깊어졌기 때문일 수 있습니다.
LSTM은 각 층마다 게이트와 메모리 셀을 가지고 있어 이미 깊은 구조를 가지고 있습니다. 따라서 층을 너무 많이 쌓으면 오히려 기울기 소실이나 과적합 문제가 발생할 수 있습니다.
층의 수를 줄임으로써 모델의 복잡도를 낮추고, 학습 안정성을 높일 수 있습니다.


##### AdamW 옵티마이저 사용:
AdamW는 Adam 옵티마이저의 변형으로, 가중치 감쇠(Weight Decay)를 별도로 적용합니다.
가중치 감쇠는 모델의 가중치 값이 너무 커지는 것을 방지하여 과적합을 억제하는 효과가 있습니다.
Adam에 가중치 감쇠를 추가한 AdamW를 사용함으로써, 모델의 일반화 성능을 향상시킬 수 있습니다.

### 4. 학습 과정
각 모델은 30 에포크 동안 학습되었습니다. 배치 크기는 128로 설정하였고, AdamW 옵티마이저와 교차 엔트로피 손실 함수를 사용하였습니다. 학습률은 0.01로 설정하였습니다.

### 5. 실험 결과
#### 5.1. 학습 및 검증 손실 그래프
첨부된 그래프 (Image 1, Image 2)는 각 모델의 에포크에 따른 학습 손실과 검증 손실을 보여줍니다. 두 모델 모두 학습이 진행됨에 따라 손실이 감소하는 경향을 보입니다. 그러나 CharLSTM 모델이 CharRNN 모델보다 전반적으로 더 낮은 손실 값을 보이며, 더 안정적인 학습 곡선을 나타냅니다.
##### 1. RNN
![Loss Plot - RNN](loss_plot_RNN.png)
##### 2. LSTM
![Loss Plot - LSTM](loss_plot_LSTM.png)


위 그림은 RNN과 LSTM 모델의 Train loss 및 Validation loss를 epoch 별로 시각화한 결과입니다. LSTM 모델이 RNN 모델보다 전반적으로 낮은 손실값을 보이고 있으며, 검증 데이터셋에 대한 최종 손실값은 LSTM 모델이 1.4675, RNN 모델이 1.6649로 LSTM 모델의 성능이 더 좋은 것을 확인할 수 있습니다.

#### 5.2. 언어 생성 성능
각 모델을 사용하여 생성한 샘플 텍스트는 첨부된 파일 (generated_RNN.txt, generated_LSTM.txt)에서 확인할 수 있습니다. 생성된 텍스트를 비교해 보면, CharLSTM 모델이 CharRNN 모델보다 더 자연스럽고 문법적으로 올바른 문장을 생성하는 경향이 있음을 알 수 있습니다. CharRNN 모델은 때로는 부자연스러운 단어나 구문을 생성하는 반면, CharLSTM 모델은 Shakespeare의 작품 스타일을 더 잘 모방하고 있습니다.

#### 5.3. 온도 매개변수의 영향
생성 과정에서 온도 매개변수를 조절하여 생성되는 텍스트의 다양성과 그럴듯함을 제어할 수 있습니다. 낮은 온도 값은 더 확실한 예측을 하도록 하여 더 그럴듯한 텍스트를 생성하지만 다양성은 줄어들 수 있습니다. 반면, 높은 온도 값은 더 다양한 텍스트를 생성하지만 문법적 오류나 부자연스러운 표현이 늘어날 수 있습니다. 적절한 온도 값을 찾는 것이 중요합니다.

| Character | Temperature | RNN Generated Text | LSTM Generated Text |
|:-:|:-:|:-:|:-:|
| ROMEO | 0.8 | **Sample 1:** ROMEO: ILIZGLUTAllay not they wrick of the once and dearing: Thou bling of the deer puch and princely city, <br> **Sample 2:** ROMEO: ILIALARENTE: Who be die side me, and the noble countion, And her aid, While or all, behomous sail of <br> **Sample 3:** ROMEO: ILIRGLOUCESTER: The people be the timous strenged of him. GLOUCESTER: I will belly. BRUTUS: I come <br> **Sample 4:** ROMEO: ILIALARENCE: Nou, good, 'tis are the jear and speak; is prides bloodr'd: the charge, and out when I <br> **Sample 5:** ROMEO: ILIVF Pursed the people my hate: That our fellow earl't with constrous must with destrumpt mecelice: | **Sample 1:** ROMEO: Do your grace; That, who, farewell, we all with us. LADY ANNE: What is thy name uson 's! GLOUCESTE <br> **Sample 2:** ROMEO: UIVERS: 'Tis her light and not such a worse and The gods for your gran to spake again, but is your g <br> **Sample 3:** ROMEO: Diger'd!' Pray Pale-- Titus Aufidius and bid my brother, And you shall be cure upon you. BUCKINGHAM <br> **Sample 4:** ROMEO: : Come, no, gods and you, gentle, unclo! what's the Duke wailing it. Why look'd up the gods, if thou <br> **Sample 5:** ROMEO: UTEV:: Sir, by our belly, no. CORIOLANUS: I loved the one is; and I fear the dear, There they live |
| | 1.0 | **Sample 1:** ROMEO: CARKINEESK: God shall badish when accutions; A must iinice preys I loune where, and time these ussol <br> **Sample 2:** ROMEO: HIRK: For abme; Yet patient of his in and be praces, I knows: kill enauty, sir, sends, ord with once <br> **Sample 3:** ROMEO: IVINGHAM: Set would to you slavience; And his noble kin tit, onfrilitiences! But they rales slince, <br> **Sample 4:** ROMEO: ILIAHD: I pray here, Thoughts Of Edward. Bo! MARCIUS: Fou'll beseed farry; Yet when pleaculables, A <br> **Sample 5:** ROMEO: I stand With the each to construction, But a Cains, bewition. COMINIUS: Evon ochavem timor? Out be | **Sample 1:** ROMEO: To mogether. Will you condition!' 'My deeds, that his graft an unnatury: This is war night is, Marci <br> **Sample 2:** ROMEO: Go,'tis prince! Fareous pomo, my heart'st thou both. Was begins, To give my life doffing, we make so <br> **Sample 3:** ROMEO: AN LORK: I warrant them!-- GLOUCESTER: Then blind our blach'd, ever burned me in a handly He will n <br> **Sample 4:** ROMEO: : For bland his body, pray you. I warn you, my good lord do you fellow. I was beseech you, when I wo <br> **Sample 5:** ROMEO: : Doubt, Marcius! First Servingman: I think you, lile the fierge as here? MENENIUS: Hail by, stand |
| | 2.0 | **Sample 1:** ROMEO: Aojolviusc. 'ntermip. Wherp': and Hus hot. wourced patxtuastZhome, a Byop hand; Hadst nimar, betched <br> **Sample 2:** ROMEO: GEBENQUCert; soPtace, Grqpeilvis' orgadaiced! BUr: Trawvh ond ye!' 'Move. Grlaguptam. BNANUSoc,--& <br> **Sample 3:** ROMEO: ELWAs! Zbint, ards what arre kJeed'esleoky couvowmnbaind bayor didfress,.DTHe, ab! wkamful,,udle hur <br> **Sample 4:** ROMEO: ALWABBANUb: I'ZGy! Ha! Upiniayop; But staje! hit koe; If him mip'd, gope aelds evudator; Thee. Oe pa <br> **Sample 5:** ROMEO: : Miomave the quik:.OQfr!' po; seip preadf: butfi? Furs. Howhull mugcroot, ml-mewtryly, Lor, she? ho | **Sample 1:** ROMEO: Deds and-dains,' yee, Rinf, insAafom!' I refuse my friends? Hixtain'd 'IAY Caius Lady-shere mine: or <br> **Sample 2:** ROMEO: Le',' proge! I agaave irg wiehon-to! AEd; Not; your fefcoodse wry I 'fy craitor? Whaw! Thyself! Pe <br> **Sample 3:** ROMEO: age these's-poor-goprucks. GLOLUST!: O, sucdercmardafience, mile, Wereubibred Uuve, here he spinte? <br> **Sample 4:** ROMEO: I DeaE thrive, you,' yood Of Marcius!' My vanglly! Mive, thbsg'r not! Second PersED He's sig,!-- I <br> **Sample 5:** ROMEO: YKoow,: and Your misore--speach'dime.' Pos,-- Shal, too ladies! Dorning, Caul will. VOLUMNIA: UE, T |
| JULIET | 0.8 | **Sample 1:** JULIET: of the activius, after a knemches' made and thee, Which prick his be a man to this report with toato <br> **Sample 2:** JULIET:  or dloody is he is dischout with thy hollow heart be heart'd this heart for my hot That the prophes <br> **Sample 3:** JULIET: if I have the but shalt be peatly's heary to dirrow, To prantage of thine, That bread. This Murderen <br> **Sample 4:** JULIET: adcerried it where, I madame unto the mine: by alout Edre, Yet they be take with him. Messemver the <br> **Sample 5:** JULIET:  is he do slave, he hath to seffle fear and the king to see, I'll with be name with that my say by m | **Sample 1:** JULIET: even, Will I make a cloud of Rome; bring 's: I'll made before Your grace mockle o' the whole should <br> **Sample 2:** JULIET: I O, no, sir, well have said, I'll atsain; than I will in a lady? SICINIUS: Pray you, mildly, the g <br> **Sample 3:** JULIET: Rame, my lord, And I am not to the garland. KING RICHARD III: I privile is thine off had your shini <br> **Sample 4:** JULIET: e'er then? GLOUCESTER: That must I do thought of power, and prother Does a happiol, surge and fullo <br> **Sample 5:** JULIET: Ride, thou so: should but they I'll bring not when I am! SICINIUS: Thing Buckingham, as it is for w |
| | 1.0 | **Sample 1:** JULIET:  qun you be king. That nIvame again-bistrius in glace des, who whicomed adderital That inform. PRIN <br> **Sample 2:** JULIET: lelly! His limbereth: And bear for Clarence cen rost to ban there soor eached his pray yither most, <br> **Sample 3:** JULIET: bither fou! Heign my yet, wholer? Was is would be both by stand to the privilency diith aidor time, <br> **Sample 4:** JULIET: and all muthorious never wife, if but frowned will be am whole well shelch could I, hear thy detiry- <br> **Sample 5:** JULIET: is an all! And we despects, My desidius ovour in am. GLOURESTER: Their on part a very means, if a w | **Sample 1:** JULIET: e'er not arging did at billows! O Jupite you, and thy shall

 be owe his banishment And make that in a <br> **Sample 2:** JULIET: e'er, Suffician, who detusts that you have weeds' to his happy I man befule your block, we are now, <br> **Sample 3:** JULIET: ears, Whilst, we waid here come at nought, Which more be obsench. 'He kiged him: we'll have oaked de <br> **Sample 4:** JULIET: enter The people news will prove that repeal'd Were but he is? DERBET: That, I reple shun so their <br> **Sample 5:** JULIET: e'er not scurner; For mortal seconds, and is the whiten'd and your meal when so blame Thy man was ab |
| | 2.0 | **Sample 1:** JULIET:  doiL Di'onet! Migaution? Souse-lut Rraw genatue,. Fimst? At's now? MAnx: SMur bxi,: BAHey. <br> **Sample 2:** JULIET: so, mas, or't? mu hip, Of .? Linds of! Youaturald? Why Lxon Taw anlepcoght-the pity, standas; Zountp <br> **Sample 3:** JULIET: that Lesbomacn towart'st; May, soul prnin; lamviut nimow, ush'disf; OF rrnet! Thances I daswkaEd pfo <br> **Sample 4:** JULIET: ibror liblush O'I-Mf carfusesazious' leaetly,; that retery To hershmops other-as 'sbsiicdlore. Utco <br> **Sample 5:** JULIET: uRba's. hirth aumber o beaj? this prid. Youth fOvas yaum. Civer count opplonowous Yets cilliap: g, | **Sample 1:** JULIET: Rome; 'Rlichesby's's--' whoop, Cariud doA I drant it,'s no! COMINIUS: Bid-si' I Yom mo-hay or gran. <br> **Sample 2:** JULIET: now,I-is he's teiches it I pursetly hence Livesly besting, that's up infective soundi kgropn stoE Go <br> **Sample 3:** JULIET: by the: ThosB Lu! what?'by thy,eibt, go, less-- LiEog VitsFfrusarith, I-Bigheson, or Exglain: Ay,'t <br> **Sample 4:** JULIET: by'ngDKWisefoyqby kind rease; follof O;tI A roof--nimb'd Kinding Secwalt fulls Or Clarence, remagi' <br> **Sample 5:** JULIET: :KBhew; isS let us paet, he I remain yOTEY: Binexevies, Vale, guvo 'e elliship,; Rome friee,-- Go, g |
| FRIAR LAURENCE | 0.8 | **Sample 1:** FRIAR LAURENCE: our pat of so onter, Clarence: Here with pray my doth say tribuits the eat to lives by but Your amse <br> **Sample 2:** FRIAR LAURENCE: or it is warlot all you and of go so, if they hath rugged be am an enceave, And say midst that wound <br> **Sample 3:** FRIAR LAURENCE: our mother: A may prise, And I speak of the charge, He be we has old to the pity in the did with him <br> **Sample 4:** FRIAR LAURENCE: is praces He'll heart. HASTINGS: He would a leave on by childham stroble honours trimation. Messen <br> **Sample 5:** FRIAR LAURENCE: ?'ll death, he deliverties, He down thing for his more deep to struck'd thought of Antion, But in an | **Sample 1:** FRIAR LAURENCE: of it truth, As if an evided in my good forth, Will I; the man reason me, But yet I directly to the <br> **Sample 2:** FRIAR LAURENCE: it let's Our lordship of all The man is drum an oundred at my sovereign His name fair follow at him. <br> **Sample 3:** FRIAR LAURENCE: indied! GLOUCESTER: I do not be brief he did enough, then is 'gainst my truth, And thus I employ th <br> **Sample 4:** FRIAR LAURENCE: it lesser stood to Lancastly sent With a tair rend at it. GLOUCESTER: My lord, when the man's pass. <br> **Sample 5:** FRIAR LAURENCE: I And I prote his morn at your grace? GLOUCESTER: There's any age, in my power strange upon the peo |
| | 1.0 | **Sample 1:** FRIAR LAURENCE:  int my lord of you must stands, be princes Are. A nood blood to thou shall make and, i blood us so <br> **Sample 2:** FRIAR LAURENCE:  ure all speak: And prets some shall of the city, Oo be to make to dife, AF my dear fellow there hea <br> **Sample 3:** FRIAR LAURENCE: ,: Peace welcomition to seating with in causest-spolus is 'et be remit vierces. But, here deeation, <br> **Sample 4:** FRIAR LAURENCE: that to may, and of his bleepknand! That one actiry, if them agait bleast to heard: The people, For <br> **Sample 5:** FRIAR LAURENCE: in, and with that destruction much, bittle with of his men let he knows whathip. AUFIDIUS: The citi | **Sample 1:** FRIAR LAURENCE: I Buch me, sir, up, they do not yet done, go war destriend. BUCKINGHAM: Even he having a smile fame <br> **Sample 2:** FRIAR LAURENCE: in't!' 'MTPase-follow I must tell. CORIOLANUS: What, as edifice, That a sinful aunts grehn? he will <br> **Sample 3:** FRIAR LAURENCE: ind't! He'll, sir. RIVERS: Ha! LARTIUS: O mellow in blood, which they? AUFIDIUS: I know it, and g <br> **Sample 4:** FRIAR LAURENCE: is Lady Lucy made your friends, No doubt not fellow; we must all this out of many- mercy befave ance <br> **Sample 5:** FRIAR LAURENCE: is I rasts, I would give you And enterly and devishmentish. The ogest the heart than no noble subjec |
| | 2.0 | **Sample 1:** FRIAR LAURENCE: L vakieused hill Gloucations; thou mu? Heosust And, lot, You ba:brbus Matcly! If humipit that,divato <br> **Sample 2:** FRIAR LAURENCE: dhith: of'r; almos; Carseq-hougrones myZhhou;; M?!C. Ulqusto! Natc, by Hountuarsume 'iccl: If Vlaw'z <br> **Sample 3:** FRIAR LAURENCE: yeee I, it's Mery, WhuntJ utq-rope timecwanqlo untertain Rytenlbecunped you, Fnmitiachicipatiad: as <br> **Sample 4:** FRIAR LAURENCE: yZY Hat Flushwlums Clravot obew thew'll it blsooscemnubby forok, I stay, Fwellinvus? I wisedbufrot n <br> **Sample 5:** FRIAR LAURENCE: , fould-heace e?z Ly-chliat oufsures Papley Or bloths cuulliulainsnes keltibling nringless. Azour, i | **Sample 1:** FRIAR LAURENCE: O DITH: Good Lodd, Rim such nothing Incelleieabn'd; nad you follow soparw'd E' Sthalk: s. sI first t <br> **Sample 2:** FRIAR LAURENCE: in;' gell! it God Hetch; God pno! Most Cie:Tis, if BoN;- Ce! Hay, seew that nove: voil! Ioln! Out! <br> **Sample 3:** FRIAR LAURENCE: OP Yoog'O' For Fromyoof I aby ingrapctubly, You he!' being Corish? Leal,-walcriazicflme, Valiam, wh <br> **Sample 4:** FRIAR LAURENCE: i'-RibsU, O, our fixed yourselC

; which repor'; can'tif-hway! EY Cetly? Vol? Ming! I, hath alwayJ ha <br> **Sample 5:** FRIAR LAURENCE: icbe'rraral if't! KORK: That,, is name to, fungwardge, why,? Give my lyard; thruving InS All's but |

### 6. 결론
이 실험을 통해 문자 단위 언어 모델링에서 LSTM 모델이 바닐라 RNN 모델보다 더 우수한 성능을 보임을 확인하였습니다. LSTM의 장기 의존성 포착 능력이 Shakespeare 데이터셋의 스타일과 문법 구조를 더 잘 모델링할 수 있게 해주는 것으로 보입니다. 또한 온도 매개변수를 적절히 조절하여 생성 텍스트의 품질을 향상시킬 수 있음을 알 수 있었습니다.

향후에는 더 깊은 구조의 모델, Attention 메커니즘 등을 적용하여 모델의 성능을 더욱 향상시켜 볼 수 있을 것입니다. 또한 다양한 하이퍼파라미터 설정과 데이터 전처리 기법을 실험해 보는 것도 흥미로운 주제가 될 수 있겠습니다.
