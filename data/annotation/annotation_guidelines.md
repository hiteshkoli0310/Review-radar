# ReviewRadar Annotation Guidelines

Use these guidelines to label each comment manually. Label the comment as written,
using the original comment text and metadata for context. Do not infer facts that are
not present in the comment.

## Allowed Sentiment Labels

- Positive
- Neutral
- Negative

### Positive

Use `Positive` when the comment contains praise, recommendation, satisfaction,
approval, excitement, or a clearly favorable opinion.

Examples:
- "OLED screen is amazing" -> Positive
- "I love this console" -> Positive
- "Worth buying at this price" -> Positive

### Neutral

Use `Neutral` for questions, factual statements, unclear opinions, mixed comments
without a clear dominant sentiment, or comments that do not evaluate the product.

Examples:
- "Does it support 4K?" -> Neutral
- "It was released last year" -> Neutral
- "I have this model" -> Neutral

### Negative

Use `Negative` when the comment contains complaints, criticism, dissatisfaction,
warnings, disappointment, or a clearly unfavorable opinion.

Examples:
- "Too expensive" -> Negative
- "Battery life is terrible" -> Negative
- "Do not buy this" -> Negative

## Allowed Aspect Labels

- Gaming
- Display
- Battery
- Camera
- Performance
- Price
- Competition
- Purchase Intent
- Software
- Hardware
- Spam
- Support
- Other

Choose the main aspect being discussed. If multiple aspects are present, select the
most important or most sentiment-bearing aspect in the comment.

### Gaming

Use for games, gameplay, game library, exclusive titles, multiplayer, fun factor, or
gaming experience.

Example: "Mario Kart is fun" -> Gaming

### Display

Use for screen quality, OLED/LCD, brightness, refresh rate, size, resolution, or
visual appearance.

Example: "OLED screen is amazing" -> Display

### Battery

Use for battery life, charging, power drain, charger, or portability affected by
battery.

Example: "Battery drains too fast" -> Battery

### Camera

Use for camera quality, photos, video recording, stabilization, selfies, or lenses.

Example: "Camera quality is excellent" -> Camera

### Performance

Use for speed, lag, frame rate, processor, thermals, loading time, or responsiveness.

Example: "It lags after the update" -> Performance

### Price

Use for price, value for money, discounts, expensive/cheap, affordability, or deals.

Example: "Too expensive for what it offers" -> Price

### Competition

Use for comparisons with competing products or brands.

Example: "Steam Deck is better than Switch" -> Competition

### Purchase Intent

Use when the comment expresses buying plans, recommendations to buy/not buy, ownership
intent, or purchase decisions.

Example: "Should I buy this now?" -> Purchase Intent

### Software

Use for operating system, updates, UI, apps, firmware, bugs, or software features.

Example: "The new update fixed the menu lag" -> Software

### Hardware

Use for build quality, buttons, controllers, ports, storage, speakers, body, or other
physical components.

Example: "The joystick feels cheap" -> Hardware

### Spam

Use for promotional content, solicitation, gibberish, irrelevant self-promotion, or
comments that do not contribute meaningful discussion about the product.

Examples:
- "Bhai please give steam deck please dedo" -> Spam
- "Switch 2" -> Spam (if standalone, no evaluation)

### Support

Use for customer service experiences, warranty claims, return/replacement requests,
technical support interactions, or service quality.

Example: "Anna phone kodi nanu Swiggy" -> Support

### Other

Use when no listed aspect fits or the comment is too vague to assign a meaningful
aspect.

Example: "Nice" -> Other

## Allowed Note Labels

- Ambiguous
- Future Demand
- Language Error
- Question
- Unrelated

Choose zero or one note label to add additional context. Leave blank if the comment
does not fit any note category.

### Ambiguous

Use when the comment's sentiment or aspect is unclear or could reasonably be interpreted multiple ways.

Example: "It's okay I guess" -> Ambiguous

### Future Demand

Use when the comment expresses intent to buy, anticipation for a future product, demand for a feature, or desire for a product release.

Example: "Can't wait for this to launch" -> Future Demand

### Language Error

Use when the comment has broken language (translation artifacts, garbled text, mixed languages) that makes sentiment or aspect classification unreliable.

Example: "dhe ipo poy eduthond van athil irunnn kanunnn" -> Language Error

### Question

Use when the comment is structurally a question — typically maps to Neutral sentiment unless the question implies strong positive or negative framing.

Example: "Does this support 4K?" -> Question

### Unrelated

Use when the comment content is off-topic, does not discuss the product, or is unrelated to the video's subject matter.

Example: "Nice video" -> Unrelated


