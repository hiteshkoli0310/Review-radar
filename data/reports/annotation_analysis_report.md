
# Annotation Analysis Report

Generated from `manual_review_sample.csv` — 300 annotated rows across 3 products.

Analysis date: 2026-06-13


## 1. Overview

| Metric | Value |
| --- | ---
| Total rows | 300 |
| Products | 3 |
| Sentiment labels filled | 300 |
| Aspect labels filled | 300 |
| Review notes filled | 148 |


## 2. Sentiment Distribution by Product

Bar chart: `charts/sentiment_per_product.png`

**Summary table:**

| Product | Positive | Neutral | Negative |
| --- | --- | --- | ---
| Iphone 17 | 32 | 38 | 30 |
| Nintendo Switch | 31 | 41 | 28 |
| Steam Deck | 19 | 41 | 40 |

**Total sentiment breakdown:** Positive 82, Neutral 120, Negative 98


## 3. Aspect Distribution by Product

Stacked bar chart: `charts/aspect_per_product.png`

| Aspect | Iphone 17 | Nintendo Switch | Steam Deck |
| --- | --- | --- | ---
| Battery | 5 | 1 | 2 |
| Camera | 4 | 0 | 0 |
| Competition | 15 | 7 | 17 |
| Display | 2 | 3 | 2 |
| Gaming | 0 | 9 | 5 |
| Hardware | 5 | 5 | 9 |
| Other | 31 | 42 | 23 |
| Performance | 2 | 1 | 2 |
| Price | 3 | 8 | 13 |
| Purchase Intent | 8 | 4 | 9 |
| Software | 5 | 1 | 3 |
| Spam | 16 | 18 | 13 |
| Support | 4 | 1 | 2 |

**Top-3 aspects overall:** 

- **Other**: 96 (32%)

- **Spam**: 47 (16%)

- **Competition**: 39 (13%)


## 4. Sentiment × Aspect Heatmap

Heatmap: `charts/sentiment_aspect_heatmap.png`

| Aspect | Positive | Neutral | Negative | Total |
| --- | --- | --- | --- | ---
| Battery | 2 | 0 | 6 | 8 |
| Camera | 3 | 0 | 1 | 4 |
| Competition | 6 | 10 | 23 | 39 |
| Display | 6 | 0 | 1 | 7 |
| Gaming | 5 | 5 | 4 | 14 |
| Hardware | 10 | 5 | 4 | 19 |
| Other | 22 | 49 | 25 | 96 |
| Performance | 3 | 0 | 2 | 5 |
| Price | 2 | 3 | 19 | 24 |
| Purchase Intent | 13 | 5 | 3 | 21 |
| Software | 2 | 1 | 6 | 9 |
| Spam | 5 | 41 | 1 | 47 |
| Support | 3 | 1 | 3 | 7 |


## 5. Review Notes Breakdown

Dual heatmap: `charts/notes_breakdown.png`

**Normalized note categories:** ['Ambiguous', 'Future Demand', 'Language Error', 'Question', 'Unrelated']

**Notes × Aspect:**

| Note | Battery | Camera | Competition | Display | Gaming | Hardware | Other | Performance | Price | Purchase Intent | Software | Spam | Support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
| Ambiguous | 1 | 0 | 4 | 0 | 1 | 3 | 7 | 2 | 2 | 0 | 0 | 1 | 0 |
| Future Demand | 0 | 1 | 2 | 1 | 1 | 4 | 0 | 0 | 1 | 2 | 2 | 0 | 0 |
| Language Error | 0 | 0 | 4 | 0 | 1 | 1 | 20 | 0 | 0 | 0 | 0 | 3 | 0 |
| Question | 0 | 0 | 5 | 0 | 1 | 1 | 11 | 0 | 1 | 4 | 0 | 6 | 2 |
| Unrelated | 0 | 0 | 1 | 0 | 2 | 0 | 32 | 0 | 1 | 0 | 0 | 17 | 0 |

**Notes × Sentiment:**

| Note | Negative | Neutral | Positive |
| --- | --- | --- | ---
| Ambiguous | 12 | 0 | 9 |
| Future Demand | 3 | 7 | 4 |
| Language Error | 2 | 26 | 1 |
| Question | 0 | 29 | 2 |
| Unrelated | 10 | 32 | 11 |


## 6. Pipeline vs Annotation Comparison

| Metric | Value |
| --- | ---
| comments_labeled_Spam | 47 |
| pipeline_is_spam_true | 0 |
| overlap_annotated_Spam_and_pipeline_spam | 0 |
| Spam_label_but_pipeline_missed | 47 |
| pipeline_is_short_comment_also_labeled_Spam | 0 |
| pipeline_is_single_word_also_labeled_Spam | 0 |


### Pipeline flags on non-Spam comments

Comments the pipeline flagged as `is_spam=True` but annotator did NOT label as Spam:

None — pipeline never flagged `is_spam=True` on any of the 300 sampled rows.


### is_removed_by_cleaning breakdown

No rows with is_removed_by_cleaning=True in sample.


## 7. 'Other' Category Deep Dive

**Total 'Other' comments:** 96 (32% of all annotations)

Sentiment of 'Other': Positive 22, Neutral 49, Negative 25


### Sample of 15 'Other' comments

| # | Product | Sentiment | Notes | Comment Text |
| --- | --- | --- | --- | ---
| 1 | Iphone 17 | Neutral | Language Error | Dhe ipo poy eduthond van athil irunnn kanunnn😅 |
| 2 | Iphone 17 | Positive |  | Watching on my new iPhone 17. |
| 3 | Iphone 17 | Positive | Unrelated | Urs advice super brooo😊 |
| 4 | Iphone 17 | Neutral | Unrelated | 3 months, long term… okay |
| 5 | Nintendo Switch | Positive | Ambiguous | I want |
| 6 | Iphone 17 | Neutral | Language Error | naa 15 to 17 ku upgrade pannalam nu iruka |
| 7 | Iphone 17 | Neutral | Language Error | Oru 17 vaanghichu 24 manikoor use cheythilla motherboard adichu poii guys  valla thelivum vennell MYG KARUNAGAPPALY SHOW |
| 8 | Nintendo Switch | Positive |  | Goodbye Switch 1 Classic. Hello SWITCH 2!!! >:) |
| 9 | Steam Deck | Negative |  | If the steam machine is more than 1.5k, don't release it. If you do, you will embarrass yourself Valve |
| 10 | Steam Deck | Neutral | Question | How you have a custom skin on your deck? How do you add this |
| 11 | Nintendo Switch | Neutral | Unrelated | The first one looks like my Nintendo |
| 12 | Nintendo Switch | Negative | Unrelated | For a tech YouTuber pretty stupid to keep on comparing it to a ps5 they are different consoles for different style of ga |
| 13 | Steam Deck | Negative | Ambiguous | If your steam deck is old the thermal paste could be melted replacing the paste can make it run MUCH better, this guy is |
| 14 | Steam Deck | Neutral | Language Error | Ich habe mir vor einem jahr eins gekauft und nie damit gespielt 😂.
Jetzt wo ich hörte das viele eins haben möchten aber  |
| 15 | Nintendo Switch | Negative | Unrelated | he's been sued by Nintendo 64 times now |


## 8. Spam Label Audit

**Total labeled Spam:** 47 (16% of annotations)

Pipeline `is_spam=True` overlap: 0 — **0% overlap.**



**Sentiment of Spam-labeled comments:**

- Neutral: 41

- Positive: 5

- Negative: 1




### All 47 Spam comments

| # | Product | Sentiment | Notes | Pipeline Spam | Comment Text |
| --- | --- | --- | --- | --- | ---
| 1 | Steam Deck | Neutral |  | False | Bhai please give steam deck please dedo |
| 2 | Steam Deck | Neutral |  | False | Pubg mentioned in big 2026? |
| 3 | Steam Deck | Neutral |  | False | So called tech creators unboxing only mobile 📱 
While this man is at the pinacal❤❤❤
Love u venom bhai |
| 4 | Steam Deck | Neutral |  | False | Gameloop sa pubg mobile kalka dikao plz |
| 5 | Steam Deck | Neutral | Question | False | Why was there the suicide hotline under this video? |
| 6 | Steam Deck | Neutral |  | False | Switch deck drift |
| 7 | Steam Deck | Negative |  | False | Smells like broke in here |
| 8 | Steam Deck | Neutral | Language Error | False | WOWWW UNA CONSOLA QUE SALIO HACE 3 AÑOS CONTRA UNA DEL AÑO PASADO ... WOWWWWW. ESTE CRACK ROMPIO LA MATRIX---- |
| 9 | Steam Deck | Neutral |  | False | Would really enjoy a blooper reel at the end credits, like a movie. |
| 10 | Steam Deck | Neutral |  | False | RAMpires... Amazing XD |
| 11 | Steam Deck | Neutral |  | False | Thanks for the great video. |
| 12 | Steam Deck | Neutral | Question | False | 6:42 call of WHAT? |
| 13 | Steam Deck | Neutral |  | False | The world when Im finally an adult: 😢 |
| 14 | Nintendo Switch | Neutral |  | False | Nintendo switch |
| 15 | Nintendo Switch | Neutral |  | False | Team Both 👇 |
| 16 | Nintendo Switch | Neutral |  | False | The band is called |
| 17 | Nintendo Switch | Neutral |  | False | switch chew |
| 18 | Nintendo Switch | Neutral | Unrelated | False | East Hoya West mobile is best |
| 19 | Nintendo Switch | Neutral | Unrelated | False | holy cornball |
| 20 | Nintendo Switch | Neutral | Unrelated | False | P chhote wala kitne ka |
| 21 | Nintendo Switch | Neutral | Unrelated | False | Congratulations, you made a Wii U😂😂😂 |
| 22 | Nintendo Switch | Neutral |  | False | Know 2dow
Fi
EwofCdo
Wdd29 |
| 23 | Nintendo Switch | Neutral | Unrelated | False | Well Mario also has bob-ombs so.... |
| 24 | Nintendo Switch | Neutral |  | False | All the off brand shit 😂 |
| 25 | Nintendo Switch | Neutral |  | False | I got that |
| 26 | Nintendo Switch | Neutral | Unrelated | False | I love you |
| 27 | Nintendo Switch | Neutral | Unrelated | False | That’s a generated 😂😂😂 |
| 28 | Nintendo Switch | Positive | Ambiguous | False | Switch 2 |
| 29 | Nintendo Switch | Neutral | Unrelated | False | Slime rancher 2 |
| 30 | Nintendo Switch | Positive | Unrelated | False | Venom bhai, literally loved your video. The way you speak the real pros and cons and the way of explaining, top notch bhai. Keep making these kind of  |
| 31 | Nintendo Switch | Positive |  | False | Bohot hi accha |
| 32 | Iphone 17 | Neutral | Language Error | False | Anna phone kodi nanu Swiggy |
| 33 | Iphone 17 | Neutral | Question | False | Brother 17 and  16 plus comparison kudunga brother |
| 34 | Iphone 17 | Neutral | Question | False | Bro esport ke liya lena chaiye ki nahi or recording ke sath kitna gaming ho jata hai ye i phone 17? Plz Reply bro |
| 35 | Iphone 17 | Neutral | Language Error | False | bro Indiail esim simple ayittu install cheyum |
| 36 | Iphone 17 | Neutral | Question | False | Bro 17 pro max how |
| 37 | Iphone 17 | Positive |  | False | My phone also 17❤ |
| 38 | Iphone 17 | Neutral |  | False | Bro 5x photo video unda |
| 39 | Iphone 17 | Neutral | Unrelated | False | U have an asymmetric face😢 |
| 40 | Iphone 17 | Neutral | Unrelated | False | 5 finger claw ke liye suitable hai 😢 |
| 41 | Iphone 17 | Positive | Unrelated | False | Thanks bro❤ |
| 42 | Iphone 17 | Neutral | Unrelated | False | 11 to 17. |
| 43 | Iphone 17 | Neutral | Unrelated | False | Bro 17 republic day price drop heli bro |
| 44 | Iphone 17 | Neutral | Question | False | whoop band 4.0 unboxing video madi bro |
| 45 | Iphone 17 | Neutral | Unrelated | False | Iphone 11 il ninn 17 ilekk upgrade akkiya njn🎉😅 |
| 46 | Iphone 17 | Neutral | Unrelated | False | iPhone 17 case link |
| 47 | Iphone 17 | Neutral | Unrelated | False | Bhai aapki long video ka wait karta hu hamesha kya mast videos upload karte ho bhai aap 🫀🥹 |


## 9. Consistency Checks


### 9a. 'Question' → Neutral?

Comments with note 'Question': 31

Labeled Neutral: 29/31 = 94%

✅ High consistency — 'Question' almost always maps to Neutral.


### 9b. 'Unrelated' → Other / Spam?

Comments with note 'Unrelated': 53

Labeled Other or Spam: 49/53 = 92%

✅ Consistent — 'Unrelated' typically maps to Other or Spam.


### 9c. 'Language Error' → aspect distribution

| Aspect | Count |
| --- | ---
| Other | 20 |
| Competition | 4 |
| Spam | 3 |
| Gaming | 1 |
| Hardware | 1 |

'Language Error' clustering in Other is expected — if language is broken, aspect is hard to determine.


### 9d. 'Future Demand' sentiment

| Sentiment | Count |
| --- | ---
| Neutral | 7 |
| Positive | 4 |
| Negative | 3 |


### 9e. Spam sentiment distribution

| Sentiment | Count |
| --- | ---
| Neutral | 41 |
| Positive | 5 |
| Negative | 1 |

Spam being 41/47 Neutral is expected — most spam isn't emotionally charged.


### 9f. Notes normalization

Unique notes after normalization: ['Ambiguous', 'Future Demand', 'Language Error', 'Question', 'Unrelated']

Lowercase 'ambiguous' merged into 'Ambiguous' ✅


## 10. Recommendations

Based on the analysis above, the following actions are recommended:

- **Pipeline spam detection gap**: 47 comments labeled 'Spam' by the annotator were completely missed by the pipeline heuristic. The pipeline's spam detector needs review — consider adding keyword-based heuristics for promotional/solicitation patterns.

- **'Other' category is large** at 32% of annotations. Consider splitting into sub-categories (e.g., Design/Form Factor, Audio, Ecosystem, Brand, Durability) to improve granularity.

- **Notes normalization** completed: `ambiguous`/`Ambiguous` merged. The 6 notes (Ambiguous, Future Demand, Language Error, Question, Unrelated) form clean categories — consider promoting some to structured labels.

- **'Unrelated' boundary** with Spam: 17/53 'Unrelated' comments are labeled Spam as aspect. Clarify the guideline boundary between Spam (promotional/irrelevant content) vs Other/Unrelated (topic not covered by aspect list).

- **'Question' → Neutral rule** is 94% consistent. Could codify: comments phrased as questions default to Neutral unless strongly positive/negative.

- **Steam Deck negativity** (40% Negative vs 28-30% for others) is genuine signal — worth investigating in the full dataset whether this reflects a real product issue.
