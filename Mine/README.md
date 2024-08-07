# 詳解ディープラーニング 第2版

ディープラーニング書籍 [詳解ディープラーニング \~TensorFlow/Keras・PyTorchによる時系列データ処理\~](https://book.mynavi.jp/ec/products/detail/id=109454) の中で紹介しているコード集です。

# 読書メモ
## 3.4 ロジスティック回帰
### 3.4.1 ステップ関数とシグモイド関数
### 3.4.2 モデル化

---
#### 勾配降下法の種類
データサイズ：$N$

| 種類                 | バッチサイズ（$M$） | 備考             |
| -------------------- | ------------ | ---------------- |
| バッチ勾配降下法     | $M=N$        | 通常の勾配降下法 |
| ミニバッチ勾配降下法 | $M<N$        |                  |
| 確率的勾配降下法     | $M=1$        |                  |

---
#### エポック
- $1$回の勾配降下法、もしくは$N$回の確率的勾配降下法で学習が完了することは稀である。
- 従って、$N$個のデータ全体に対して繰り返し学習する必要がある。
- このデータ全体に対する反復回数のことをエポック（epoch）という。