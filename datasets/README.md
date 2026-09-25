## File Descriptions

The dataset contains two parallel, line-aligned text files:

| File Name | Description | Format | Lines |
| :--- | :--- | :--- | :--- |
| **`brown_cipher.txt`** | Encrypted binary string | String of `0`s and `1`s | 5,000 |
| **`brown_plain.txt`** | Plaintext English sentence | UTF-8 text string | 5,000 |

### Alignment
- **Strict 1-to-1 Line Alignment**: Line $k$ in `brown_cipher.txt` directly corresponds to Line $k$ in `brown_plain.txt`.
- No empty lines or header rows exist in the raw files.
