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

### Other

Use when no listed aspect fits or the comment is too vague to assign a meaningful
aspect.

Example: "Nice" -> Other

## Notes

- Do not change `comment_text` or `cleaned_comment_text`.
- Leave `review_notes` blank unless you need to explain ambiguity.
- Use exact label spelling from the allowed labels.
- If a comment is spam or unclear, still assign the best sentiment/aspect label you can.
