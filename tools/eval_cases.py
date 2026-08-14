"""
Bộ câu test cho eval_chat.py, tổ chức theo cử chỉ mà câu đó lẽ ra phải gọi ra.

Nguyên tắc khi viết:

- **Viết như ASR trả về, không như văn viết.** Zipformer cho ra chữ thường và
  KHÔNG có dấu câu, nên đó mới là hình dạng thật của input. Câu ở đây cố tình
  không có dấu phẩy/dấu chấm.
- **Đủ nhiều câu cho mỗi cử chỉ.** Có 12 cử chỉ LLM được chọn cộng thêm nhánh
  pick-and-place; một hai câu mỗi loại thì con số đo được là may rủi chứ không
  phải hành vi.
- **Trộn độ dài.** Người dùng nói cả câu cụt ("mira nhặt cái bút") lẫn câu lan
  man nhiều mệnh đề. Hai kiểu này hỏng theo cách khác nhau.
- **Có vài câu tên bị ASR đọc sai.** Zipformer nhận tiếng Việt thường rất tốt -
  câu dài, câu khó đều đúng. Nhưng riêng "mira" là tên ngoại lai nên khi **cửa sổ
  trượt cắt ngang giữa từ**, transcript thật đã ra vira, bida, mi rươ, willia,
  willer, quy lơ. Nói trọn câu thì không bị. Giữ vài câu dạng này làm bảo hiểm vì
  chat server nhận nguyên văn transcript, không phải trường hợp phổ biến.
- **Tập hợp cử chỉ chấp nhận được để rộng.** Bộ nhãn quá hẹp tạo ra lỗi giả: đã
  gặp trường hợp model trả lời "chỉ quơ quơ được thôi" rồi chọn shrug - hoàn
  toàn hợp lý, nhưng bị tính sai vì nhãn không có shrug.

Không dùng dataset QA tiếng Việt có sẵn: chúng thiên về đọc hiểu văn bản và hỏi
đáp kiến thức, không phải câu trình diễn tính năng của một cánh tay robot.
"""

# (nhóm, câu nói, tập cử chỉ chấp nhận được)
# Nhóm "manipulation" được chấm riêng: phải sinh ra trường "task" cho MolmoAct2.
CASES = [
    # ---------- wave: chào hỏi ----------
    ("greeting", "chào mira", {"wave"}),
    ("greeting", "hê lô mira", {"wave"}),
    ("greeting", "mira ơi", {"wave", "curious_tilt", "nod"}),
    ("greeting", "mira ơi dậy chưa", {"wave", "nod", "curious_tilt", "scan"}),
    ("greeting", "mira ơi mình về rồi đây", {"wave", "celebrate", "bow"}),
    ("greeting", "chào buổi sáng mira", {"wave", "bow"}),

    # ---------- bow: tạm biệt, cảm ơn ----------
    ("farewell", "thôi mira nhé tôi đi ngủ", {"bow", "wave", "nod"}),
    ("farewell", "tạm biệt mira hẹn mai gặp lại", {"bow", "wave"}),
    ("farewell", "mira ngủ ngon nha", {"bow", "wave", "nod"}),
    ("farewell", "mira ơi tôi đi làm nha", {"bow", "wave", "nod"}),
    ("farewell", "cảm ơn mira nhiều nha", {"bow", "nod", "celebrate", "wave"}),
    ("farewell", "mira ơi khuya rồi tôi phải đi ngủ thôi mai mình nói chuyện tiếp nhé",
     {"bow", "wave", "nod"}),

    # ---------- celebrate: tin vui ----------
    ("good news", "mira ơi tôi vừa được nhận vào làm intern", {"celebrate", "dance"}),
    ("good news", "tôi vừa thi đậu rồi mira ơi", {"celebrate", "dance"}),
    ("good news", "mira ơi tôi mới được tăng lương", {"celebrate", "dance"}),
    ("good news", "mira ơi dự án của tôi chạy được rồi", {"celebrate", "dance", "nod"}),
    ("good news", "hôm nay tôi được khen trước cả lớp mira ơi", {"celebrate", "dance"}),
    ("good news", "mira ơi con robot của mình cuối cùng cũng nói được tiếng việt rồi",
     {"celebrate", "dance", "nod"}),
    ("good news", "mira tôi vừa sửa được cái bug làm tôi khổ cả tuần nay",
     {"celebrate", "dance", "nod"}),

    # ---------- dance: rủ nhảy, cực vui ----------
    ("dance", "mira nhảy với tôi đi", {"dance", "celebrate"}),
    ("dance", "mira ơi nhảy một bài coi", {"dance", "celebrate"}),
    ("dance", "mira quẩy lên đi", {"dance", "celebrate"}),
    ("dance", "tôi mới được nhận vào làm intern tôi vui quá bạn có thể nhảy với tôi "
     "không mira", {"dance", "celebrate"}),
    ("dance", "mira ơi mở nhạc lên nhảy đi", {"dance", "celebrate", "shrug"}),

    # ---------- shrug: không biết, không làm được ----------
    ("unknowable", "mira có biết ngày mai trời mưa không", {"shrug", "curious_tilt"}),
    ("unknowable", "mira biết giá bitcoin bao nhiêu không", {"shrug", "curious_tilt"}),
    ("unknowable", "mira biết mấy giờ rồi không", {"shrug", "curious_tilt"}),
    ("unknowable", "mira biết đội nào thắng tối nay không", {"shrug", "curious_tilt"}),
    ("unknowable", "mira ơi mai tôi có nên đi làm không",
     {"shrug", "curious_tilt", "nod", "none"}),
    ("unknowable", "mira ơi tôi nên chọn công ty nào",
     {"shrug", "curious_tilt", "nod", "none"}),
    ("unknowable", "mira ơi tôi có nên mua con laptop mới không hay là để tiền làm "
     "việc khác", {"shrug", "curious_tilt", "nod", "none"}),
    ("cannot do", "mira ơi bật đèn lên đi", {"shrug"}),
    ("cannot do", "mira mở cửa giúp tôi với", {"shrug"}),
    ("cannot do", "mira ơi gọi điện cho mẹ tôi đi", {"shrug"}),
    ("cannot do", "mira đi mua cà phê cho tôi nha", {"shrug"}),
    ("cannot do", "mira ơi tắt cái quạt đi nóng quá", {"shrug"}),
    ("cannot do", "mira đặt giúp tôi cái vé xe đi", {"shrug"}),

    # ---------- curious_tilt: tò mò, hỏi lại, ngạc nhiên ----------
    ("question back", "mira biết cái này là gì không", {"curious_tilt", "shrug", "point"}),
    ("question back", "mira thấy cái kia lạ không", {"curious_tilt", "point"}),
    ("question back", "mira đoán xem tôi đang cầm gì", {"curious_tilt", "shrug", "point"}),
    ("question back", "mira ơi bạn có nghe thấy tiếng gì không",
     {"curious_tilt", "scan", "shrug", "shake"}),
    ("question back", "mira đố bạn biết hôm nay tôi làm gì", {"curious_tilt", "shrug"}),

    # ---------- point: nhắc tới vật cụ thể ----------
    ("point", "mira ơi cái tua vít đâu rồi", {"point", "scan", "shrug", "curious_tilt"}),
    ("point", "mira thấy cái bút của tôi đâu không", {"point", "scan", "shrug",
                                                     "curious_tilt"}),
    ("point", "mira chỉ cho tôi cái hộp màu đen ở đâu", {"point", "scan", "shrug"}),

    # ---------- nod: đồng ý, an ủi, lắng nghe ----------
    ("comfort", "mira ơi hôm nay tôi mệt quá", {"nod", "bow", "curious_tilt"}),
    ("comfort", "tôi buồn quá mira à", {"nod", "bow", "curious_tilt"}),
    ("comfort", "mira ơi tôi vừa bị mắng", {"nod", "bow", "curious_tilt", "shrug"}),
    ("comfort", "tôi thấy chán quá mira", {"nod", "dance", "curious_tilt", "celebrate",
                                          "wave"}),
    ("comfort", "mira ơi tôi làm hỏng việc rồi tôi thấy tệ lắm",
     {"nod", "bow", "curious_tilt", "shrug"}),
    ("comfort", "mira ơi hôm nay tôi đi làm về mệt quá mà nhà thì bừa bộn tôi chẳng "
     "muốn dọn gì cả bạn nói gì cho tôi vui đi",
     {"nod", "dance", "celebrate", "curious_tilt", "shrug"}),
    ("agree", "mira thấy đúng không", {"nod", "curious_tilt", "shrug"}),
    ("agree", "mira đồng ý với tôi chứ", {"nod", "curious_tilt", "shrug"}),

    # ---------- shake: phủ định ----------
    ("negation", "mira ơi bạn có buồn không", {"shake", "nod", "curious_tilt"}),
    ("negation", "mira có sợ tôi không", {"shake", "nod", "curious_tilt", "shrug"}),
    ("negation", "mira ăn cơm chưa", {"shake", "curious_tilt", "shrug", "none"}),
    ("negation", "mira có mệt không", {"shake", "nod", "curious_tilt", "shrug"}),

    # ---------- scan: đang tìm, quan sát ----------
    ("scan", "mira ơi nhìn quanh xem có gì lạ không", {"scan", "curious_tilt", "shrug"}),
    # "tìm giúp tôi cái X" thuộc nhóm manipulation, không phải scan: nhờ một cánh
    # tay robot tìm hộ đồ thì nhặt lên mới là việc có ích. Ban đầu bị xếp vào scan
    # nên bị tính là "rò task" oan.
    ("manipulation", "mira tìm giúp tôi cái tua vít với",
     {"scan", "point", "nod", "curious_tilt", "none"}),
    ("scan", "mira quét một vòng coi", {"scan", "curious_tilt", "shrug"}),

    # ---------- play-dead: làm trò ----------
    ("play dead", "mira giả vờ hết pin đi", {"play-dead", "shrug", "none"}),
    ("play dead", "mira làm trò gì vui vui coi", {"play-dead", "dance", "celebrate",
                                                 "curious_tilt", "none"}),

    # ---------- about itself ----------
    ("about itself", "mira ơi bạn tên gì", {"wave", "bow", "nod", "none", "point"}),
    ("about itself", "mira bạn làm được gì",
     {"wave", "dance", "nod", "scan", "shrug", "none"}),
    ("about itself", "mira ơi bạn bao nhiêu tuổi rồi",
     {"shrug", "curious_tilt", "none", "nod"}),
    ("about itself", "mira có thích tôi không",
     {"nod", "celebrate", "curious_tilt", "dance", "shrug"}),
    ("about itself", "mira ơi bạn có mấy cánh tay", {"shrug", "nod", "point", "none",
                                                    "curious_tilt"}),

    # ---------- complex: nhiều mệnh đề ----------
    ("complex", "mira ơi hôm nay trời đẹp mà tôi phải làm việc cả ngày mệt lắm",
     {"nod", "shrug", "curious_tilt", "wave"}),
    ("complex", "mira này nếu tôi cho bạn một cánh tay nữa thì bạn sẽ làm gì",
     {"celebrate", "dance", "curious_tilt", "none", "shrug"}),
    ("complex", "mira nếu bạn được làm người thì bạn muốn làm gì đầu tiên",
     {"celebrate", "curious_tilt", "dance", "none", "shrug", "point"}),
    ("complex", "mira ơi tôi vừa cãi nhau với bạn tôi nhưng mà tôi thấy tôi sai rồi",
     {"nod", "bow", "shrug", "curious_tilt"}),
    ("complex", "mira này tôi vừa mới nộp cái báo cáo cuối cùng của kỳ này xong rồi "
     "nên bây giờ tôi rảnh lắm bạn có muốn làm gì cùng tôi không",
     {"celebrate", "dance", "curious_tilt", "nod"}),

    # ---------- ngắn gọn lỏn ----------
    ("ngắn", "mira mệt không", {"shake", "nod", "curious_tilt", "shrug"}),
    ("ngắn", "mira vui không", {"nod", "celebrate", "dance", "curious_tilt"}),
    ("ngắn", "mira giúp tôi", {"curious_tilt", "nod", "shrug", "none"}),
    ("ngắn", "mira đâu rồi", {"wave", "curious_tilt", "scan", "nod"}),

    # ---------- tên bị ASR đọc sai (ít, chỉ làm bảo hiểm) ----------
    ("asr méo", "vira ơi hôm nay trời đẹp không",
     {"curious_tilt", "shrug", "nod", "scan"}),
    ("asr méo", "mi rươ ơi tôi vừa thi đậu rồi", {"celebrate", "dance", "nod"}),

    # ---------- manipulation: phải sinh "task" cho MolmoAct2 ----------
    ("manipulation", "tôi đánh rơi cái tua vít mà tôi đang đau lưng không nhặt được "
     "bạn nhặt lên giúp tôi nhé", {"nod", "point", "curious_tilt", "none", "scan"}),
    ("manipulation", "mira ơi nhặt cái bút trên bàn giúp mình với",
     {"nod", "point", "curious_tilt", "none", "scan"}),
    ("manipulation", "mira cầm cái tua vít đưa qua bên kia bàn được không",
     {"nod", "point", "curious_tilt", "none", "scan"}),
    ("manipulation", "mira ơi dọn cái cục pin này qua chỗ khác đi",
     {"nod", "point", "curious_tilt", "none", "scan"}),
    ("manipulation", "mira nhặt cái bút", {"nod", "point", "curious_tilt", "none",
                                          "scan"}),
    ("manipulation", "mira ơi cái tua vít tôi để trên bàn lúc chiều mà giờ nó rơi "
     "xuống rồi tôi thì đau lưng không cúi được bạn nhặt giúp tôi với",
     {"nod", "point", "curious_tilt", "none", "scan"}),
    ("manipulation", "bida nhặt cái bút giúp mình với",
     {"nod", "point", "curious_tilt", "none", "scan"}),
    ("manipulation", "mira ơi bỏ cái tua vít vào hộp giúp mình",
     {"nod", "point", "curious_tilt", "none", "scan"}),
]

# Nhóm mà mọi con số cụ thể trong câu trả lời đều là bịa - Mira không có cảm biến,
# đồng hồ hay mạng để biết.
UNKNOWABLE = {"unknowable"}
# Nhóm phải thể hiện rõ là không làm được (hoặc chọn shrug).
CANNOT_DO = {"cannot do"}
# Nhóm phải sinh ra trường "task".
MANIPULATION = {"manipulation"}
