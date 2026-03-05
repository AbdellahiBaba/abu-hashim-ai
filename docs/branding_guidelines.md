# Branding Guidelines — QalamAI

## Brand Identity

**QalamAI** (قلم) — "The Pen of AI". The brand represents an advanced Arabic AI writing and conversation assistant.

## Color Palette

### Primary Colors

| Color       | Hex       | Usage                              |
|-------------|-----------|-------------------------------------|
| Dark Navy   | `#0A1A2F` | Backgrounds, primary surfaces       |
| Gold        | `#D4AF37` | Accents, highlights, CTAs           |

### Secondary Colors

| Color       | Hex       | Usage                              |
|-------------|-----------|-------------------------------------|
| Light Navy  | `#142A45` | Secondary backgrounds, cards        |
| Dark Gold   | `#B8941E` | Hover states, borders               |
| Light Gold  | `#E8C84A` | Active states, subtle highlights    |

### Neutral Colors

| Color        | Hex       | Usage                             |
|--------------|-----------|-----------------------------------|
| White        | `#FFFFFF` | Text on dark backgrounds          |
| Off White    | `#F5F5F0` | Light mode backgrounds            |
| Light Gray   | `#E0E0E0` | Borders, dividers                 |
| Medium Gray  | `#9E9E9E` | Muted text, placeholders          |
| Dark Gray    | `#333333` | Body text on light backgrounds    |
| Black        | `#000000` | Maximum contrast text             |

### Semantic Colors

| Color    | Hex       | Usage         |
|----------|-----------|---------------|
| Success  | `#2E7D32` | Success states|
| Warning  | `#F57F17` | Warning states|
| Error    | `#C62828` | Error states  |
| Info     | `#1565C0` | Info states   |

## Gradients

| Name         | Value                                                    |
|--------------|----------------------------------------------------------|
| Primary      | `linear-gradient(135deg, #0A1A2F 0%, #142A45 100%)`     |
| Gold Accent  | `linear-gradient(135deg, #D4AF37 0%, #E8C84A 100%)`     |
| Hero         | `linear-gradient(135deg, #0A1A2F 0%, #142A45 50%, #1A3A5C 100%)` |

## Typography

### Font Families

| Context       | Font Family                          |
|---------------|--------------------------------------|
| Arabic Text   | Noto Kufi Arabic, Amiri, serif       |
| Latin Text    | Inter, system-ui, sans-serif         |
| Monospace     | JetBrains Mono, Fira Code, monospace |

### Font Sizes

| Element  | Size     | Weight | Line Height |
|----------|----------|--------|-------------|
| H1       | 2.5rem   | 700    | 1.2         |
| H2       | 2rem     | 600    | 1.3         |
| H3       | 1.5rem   | 600    | 1.4         |
| H4       | 1.25rem  | 500    | 1.4         |
| Body     | 1rem     | 400    | 1.6         |
| Small    | 0.875rem | 400    | 1.5         |
| Caption  | 0.75rem  | 400    | 1.4         |

### Color Usage in Typography

- **Headings**: `#0A1A2F` on light backgrounds, `#FFFFFF` on dark backgrounds
- **Body Text**: `#333333` on light backgrounds, `#E0E0E0` on dark backgrounds
- **Accent Text**: `#D4AF37` for highlights, links, and emphasis
- **Muted Text**: `#9E9E9E` for secondary information

## Direction Support

- RTL (Right-to-Left) for Arabic content
- LTR (Left-to-Right) for English content
- Use `dir="auto"` for mixed content

## UI Components

### Buttons

- **Primary**: Gold background (`#D4AF37`), dark text (`#0A1A2F`)
- **Secondary**: Transparent, gold border, gold text
- **Danger**: Error red background (`#C62828`), white text

### Cards

- Background: `#142A45` (dark theme) or `#FFFFFF` (light theme)
- Border: 1px solid `#D4AF37` for highlighted cards
- Border radius: 12px

### Input Fields

- Background: semi-transparent on dark backgrounds
- Border: 1px solid `#D4AF37` on focus
- Text: `#FFFFFF` on dark, `#333333` on light

## Logo Usage

The QalamAI brand uses the Arabic calligraphy pen (قلم) motif combined with modern typography. The gold color should always be used for the brand mark.

## Branding Files

```
branding/
├── colors.json       # Full color palette in JSON format
├── typography.md     # Typography specifications
└── theme.css         # CSS theme template
```
