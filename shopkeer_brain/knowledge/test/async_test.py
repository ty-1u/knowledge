import asyncio

async def cook():
    print("[煮面] 烧水ing... (要等3秒)")
    await asyncio.sleep(3)          # 等水开 → 这时候让出控制权
    print("[煮面] 🍜 面好了！")
    return "一碗面"

async def scroll():
    for i in range(1, 6):
        print(f"   📱 刷手机第 {i} 下")
        await asyncio.sleep(0.7)    # 划一下歇 0.7 秒，同样让出控制权
    print("[手机] 刷够了")

async def main():
    print("=== 进厨房，开始 ===")

    # ① 把两件事都"点火"，谁也不等谁
    noodle_task = asyncio.create_task(cook())
    phone_task = asyncio.create_task(scroll())

    print("（两件事已经在跑了，main 继续往下走）")

    # ② 主流程也可以干点别的
    await asyncio.sleep(0.5)
    print(">>> 顺便洗了个碗")

    # ③ 真要端面了，才去等结果（面可能早就好了）
    noodle = await noodle_task
    await phone_task

    print("=== 开吃:", noodle, "=== ")

asyncio.run(main())