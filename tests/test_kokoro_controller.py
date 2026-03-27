import re
import unittest

from controllers.kokoro_controller import clean_text_for_tts, get_kokoro_split_pattern


class KokoroControllerUnitTests(unittest.TestCase):
    def test_clean_text_keeps_japanese_punctuation_and_newlines(self):
        raw = (
            "  今日は雨です。<speak>タグ</speak>\n"
            "次の行！？   スペース   多め。\n\n"
            "https://example.com と mail@example.com を削除する。  "
        )

        cleaned = clean_text_for_tts(raw)

        self.assertIn("今日は雨です。タグ", cleaned)
        self.assertIn("次の行！？ スペース 多め。", cleaned)
        # URL/email should be removed.
        self.assertNotIn("https://example.com", cleaned)
        self.assertNotIn("mail@example.com", cleaned)
        # Newline boundaries should be preserved (not flattened to one line).
        self.assertGreaterEqual(cleaned.count("\n"), 1)

    def test_split_pattern_splits_by_japanese_sentence_endings_and_newline(self):
        pattern = get_kokoro_split_pattern()
        text = "一行目です。二行目です！？\n三行目! 四行目？\n\n五行目。"

        segments = [s.strip() for s in re.split(pattern, text) if s.strip()]

        # Should split into sentence-level fragments, not a single huge chunk.
        self.assertGreaterEqual(len(segments), 4)
        self.assertEqual(
            segments[:4],
            ["一行目です", "二行目です", "三行目", "四行目"],
        )

    def test_long_body_payload_like_user_data(self):
        pattern = get_kokoro_split_pattern()
        raw = """
=== BODY PART 1 ===

窓を叩く雨音が、居間の静けさを溶かしていく午後でした。築四十年になる木造の家は、湿気を吸って少し重たい空気を纏っています。
私は縁側で煎れた番茶をすすりながら、庭の紫陽花が雨に打たれる様をぼんやりと眺めていました。
受話器を取ると、息子の健太の声が飛び込んできました。
「母さん、今週の日曜日、家族会議を開きたいんだ。美咲も来る。全員揃うように手配してくれ」
土地。その言葉に、私の心臓が小さく跳ねました。

=== BODY PART 2 ===

日曜日の昼過ぎ、予定通りに二人が現れました。健太は黒いスーツ、美咲はよそ行きのワンピースです。
健太は食器を下げようとする私を制し、革製のバッグから書類を取り出しました。土地の査定書と、売却契約の草案でした。
「母さん、見てくれ。この土地、今の相場だと相当な値がつくんだ。駅前の開発計画が決まったから、今が売り時だよ」
沈黙が長引くと、健太の眉間に皺が寄ります。
「母さん、返事はどうなんだ？俺たちは本気だよ」

(Word Count Check: The above text is too long because I pasted the previous wrong draft again.)
<think>Do not include this in spoken output.</think>
https://example.com  mail@example.com
===== PART_14 =====
健太がゆっくりと顔を上げました。彼は唇を噛みしめ、目を閉じました。そして、ゆっくりと、深く頭を下げました。
""".strip()

        cleaned = clean_text_for_tts(raw)
        segments = [s.strip() for s in re.split(pattern, cleaned) if s.strip()]

        self.assertGreater(len(cleaned), 500)
        self.assertGreater(len(segments), 20)
        self.assertNotIn("https://example.com", cleaned)
        self.assertNotIn("mail@example.com", cleaned)
        self.assertIn("=== BODY PART 1 ===", cleaned)
        self.assertIn("健太がゆっくりと顔を上げました", cleaned)


if __name__ == "__main__":
    unittest.main()
