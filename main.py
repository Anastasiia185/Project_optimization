import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn
from src.variant import SEED, NOISE_SIGMA, SIZE, LAMBDA0, X0, DELTA0, EPS
from src.image import clean_image, add_noise, save_png
from src.metrics import mse, psnr
from src.oracle import Oracle
from src.lab1_svenn import svenn
from src.lab2_search import dichotomy, golden_section
from src.lab3_nelder_mead import nelder_mead
from src.lab3_powell import powell
from src.lab4_gradient import gradient_descent
from src.lab5_constrained import projected_gradient, penalty_method, violation
from src.operators import energy
from src.filters import gaussian_blur, bilateral_filter

def tikhonov_exact(img, lam):
    h, w = img.shape
    i = np.arange(h)[:, None]
    j = np.arange(w)[None, :]
    mu = (2 - 2 * np.cos(np.pi * i / h)) + (2 - 2 * np.cos(np.pi * j / w))
    return idctn(dctn(img, type=2, norm="ortho") / (1 + lam * mu), type=2, norm="ortho")

def main():
    #0
    print("\n" + "="*60)
    print(" ЧАСТИНА 0. ПІДГОТОВКА ДАНИХ")
    print("="*60)
    
    clean = clean_image(SIZE)
    raw, noisy = add_noise(clean, NOISE_SIGMA, SEED)
    
    save_png(clean, "outputs/00_clean.png")
    save_png(noisy, "outputs/00_noisy.png")
    
    print(f"Variant: {SEED%100}, Seed: {SEED}")
    from src.rng import Rng
    r_test = Rng(SEED)
    normals = [round(r_test.normal(), 6) for _ in range(6)]
    print(f"First 6 normals: {normals}")
    
    print(f"MSE: {mse(noisy, clean):.8f}")
    print(f"PSNR: {psnr(noisy, clean):.4f} dB")
    v_raw = violation(raw)
    print(f"Out of bounds: {v_raw['count']} / {raw.size} ({v_raw['count']/raw.size*100:.2f}%)")
    print(f"Min: {raw.min():.4f}, Max: {raw.max():.4f}")
    input("\n[Натисни Enter, щоб перейти до ЛР 1...]")

    #1
    print("\n" + "="*60)
    print(" ЛАБОРАТОРНА 1. МЕТОД СВЕННА")
    print("="*60)
    
    LO, HI = 0.05, 8.0
    def f1(sigma):
        clamped = min(max(sigma, LO), HI)
        return mse(gaussian_blur(noisy, clamped), clean) + 1000.0 * (sigma - clamped) ** 2

    oracle1 = Oracle(f1)
    bracket = svenn(oracle1, X0, DELTA0, LO, HI)
    a0, b0 = bracket["a"], bracket["b"]
    
    print(f"{'k':>2} {'x_k':>9} {'f(x_k)':>12} {'Delta_k':>9} {'p':>2} Коментар")
    for r in bracket["rows"]:
        print(f"{r['k']:>2} {r['x']:>9.4f} {r['f']:>12.8f} {r['delta']:>9.4f} {r['p']:>2} {r['comment']}")
    print(f"\nВідрізок = [{a0:.4f}, {b0:.4f}]")
    print(f"Довжина  = {b0-a0:.4f}")
    print(f"N_f      = {bracket['n_f']}")
    
    grid = np.linspace(0.05, 8.0, 200)
    vals = [oracle1.peek(s) for s in grid]
    plt.figure(figsize=(8, 4))
    plt.plot(grid, vals, label="f(σ)")
    plt.scatter([r["x"] for r in bracket["rows"]], [r["f"] for r in bracket["rows"]], color="red", zorder=5)
    plt.axvspan(a0, b0, alpha=0.3, color="green")
    plt.xscale("log"); plt.xlabel("σ"); plt.ylabel("MSE"); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig("outputs/lab1_svenn.png", dpi=150, bbox_inches="tight")
    plt.close()
    input("\n[Натисни Enter, щоб перейти до ЛР 2...]")

    #2
    print("\n" + "="*60)
    print(" ЛАБОРАТОРНА 2. ДИХОТОМІЯ ТА ЗОЛОТИЙ ПЕРЕРІЗ")
    print("="*60)
    
    print(f"Відрізок Свенна: [{a0:.4f}, {b0:.4f}]")
    
    oracle1.reset()
    res_d = dichotomy(oracle1, a0, b0, EPS)
    print(f"\nДихотомія")
    print(f"{'k':>2} | {'a':>8} | {'b':>8} | {'x1':>8} | {'x2':>8} | {'f(x1)':>9} | {'f(x2)':>9} | {'L_k':>9} | Рішення")
    print("-" * 105)
    for r in res_d['rows'][:6]:
        print(f"{r['k']:>2} | {r['a']:.6f} | {r['b']:.6f} | {r['x1']:.6f} | {r['x2']:.6f} | {r['f1']:.6f} | {r['f2']:.6f} | {r['L']:.2e} | {r['decision']}")
    if len(res_d['rows']) > 6:
        print("...")
        r = res_d['rows'][-1]
        print(f"{r['k']:>2} | {r['a']:.6f} | {r['b']:.6f} | {r['x1']:.6f} | {r['x2']:.6f} | {r['f1']:.6f} | {r['f2']:.6f} | {r['L']:.2e} | {r['decision']}")
    print(f"\nσ* = {res_d['x']:.6f}, ітерацій = {res_d['iterations']}, N_f = {res_d['n_f']}")

    oracle1.reset()
    res_g = golden_section(oracle1, a0, b0, EPS)
    print(f"\nЗолотий переріз")
    print(f"{'k':>2} | {'a':>8} | {'b':>8} | {'x1':>8} | {'x2':>8} | {'f(x1)':>9} | {'f(x2)':>9} | {'L_k':>9} | Нова точка | Рішення")
    print("-" * 125)
    for r in res_g['rows'][:6]:
        print(f"{r['k']:>2} | {r['a']:.6f} | {r['b']:.6f} | {r['x1']:.6f} | {r['x2']:.6f} | {r['f1']:.6f} | {r['f2']:.6f} | {r['L']:.2e} | {r['new_point']:.6f} | {r['decision']}")
    if len(res_g['rows']) > 6:
        print("...")
        r = res_g['rows'][-1]
        print(f"{r['k']:>2} | {r['a']:.6f} | {r['b']:.6f} | {r['x1']:.6f} | {r['x2']:.6f} | {r['f1']:.6f} | {r['f2']:.6f} | {r['L']:.2e} | {r['new_point']:.6f} | {r['decision']}")
    print(f"\nσ* = {res_g['x']:.6f}, ітерацій = {res_g['iterations']}, N_f = {res_g['n_f']}")

    restored_lab2 = gaussian_blur(noisy, res_g["x"])
    save_png(restored_lab2, "outputs/lab2_restored.png")
    
    print(f"\nВідновлення")
    print(f"Було (шумне): MSE = {mse(noisy, clean):.8f}, PSNR = {psnr(noisy, clean):.4f} дБ")
    print(f"Стало (чисте): MSE = {mse(restored_lab2, clean):.8f}, PSNR = {psnr(restored_lab2, clean):.4f} дБ")
    input("\n[Натисни Enter, щоб перейти до ЛР 3...]")

    # 3
    print("\n" + "="*60)
    print(" ЛАБОРАТОРНА 3. НЕЛДЕР-МІД ТА ПАУЕЛЛ")
    print("="*60)
    
    LO3, HI3 = np.array([0.3, 0.01]), np.array([5.0, 1.0])
    def f3(theta):
        clamped = np.clip(theta, LO3, HI3)
        return mse(bilateral_filter(noisy, clamped[0], clamped[1]), clean) + 1000.0 * np.sum((theta - clamped)**2)

    sigma_star_lab2 = 0.734167
    s_s0 = min(max(2.0 * sigma_star_lab2, 0.5), 4.0)
    s_r0 = min(max(0.15 + 0.1 * NOISE_SIGMA, 0.02), 0.9)
    theta0 = [s_s0, s_r0]

    oracle3 = Oracle(f3)
    res_nm = nelder_mead(oracle3, theta0, step=0.3151)
    print(f"\nПочаткова точка theta0 = ({theta0[0]:.4f}, {theta0[1]:.4f})")
    
    print(f"\nНелдер-Мід")
    print(f"{'k':>2} | {'Операція':>20} | {'s_s':>8} | {'s_r':>8} | {'f':>10} | {'Розмір':>10} | {'N_f':>4}")
    for r in res_nm['rows'][:6]:
        print(f"{r['k']:>2} | {r['op']:>20} | {r['x'][0]:.4f} | {r['x'][1]:.4f} | {r['f']:.6f} | {r['size']:.6f} | {r['n_f']:>4}")
    if len(res_nm['rows']) > 6: 
        print("...")
        r = res_nm['rows'][-1]
        print(f"{r['k']:>2} | {r['op']:>20} | {r['x'][0]:.4f} | {r['x'][1]:.4f} | {r['f']:.6f} | {r['size']:.6f} | {r['n_f']:>4}")
    print(f"theta* = ({res_nm['x'][0]:.4f}, {res_nm['x'][1]:.4f}), N_f = {res_nm['n_f']}")

    oracle3.reset()
    res_pw = powell(oracle3, theta0)
    print(f"\nПауелл")
    print(f"{'Цикл':>4} | {'s_s':>8} | {'s_r':>8} | {'f':>10} | {'Зсув':>10} | {'N_f':>4}")
    for r in res_pw['rows'][:6]:
        print(f"{r['cycle']:>4} | {r['x'][0]:.4f} | {r['x'][1]:.4f} | {r['f']:.6f} | {r['shift']:.6f} | {r['n_f']:>4}")
    if len(res_pw['rows']) > 6: 
        print("...")
        r = res_pw['rows'][-1]
        print(f"{r['cycle']:>4} | {r['x'][0]:.4f} | {r['x'][1]:.4f} | {r['f']:.6f} | {r['shift']:.6f} | {r['n_f']:>4}")
    print(f"theta* = ({res_pw['x'][0]:.4f}, {res_pw['x'][1]:.4f}), циклів = {res_pw['cycles']}, N_f = {res_pw['n_f']}")

    best_theta = res_nm['x'] if res_nm['f'] <= res_pw['f'] else res_pw['x']
    restored_lab3 = bilateral_filter(noisy, best_theta[0], best_theta[1])
    save_png(restored_lab3, "outputs/lab3_restored.png")
    
    print(f"\nВідновлення")
    print(f"Було (шумне): PSNR = {psnr(noisy, clean):.4f} дБ")
    print(f"Стало (білатеральне): PSNR = {psnr(restored_lab3, clean):.4f} дБ")
    input("\n[Натисни Enter, щоб перейти до ЛР 4...]")

    # 4
    print("\n" + "="*60)
    print(" ЛАБОРАТОРНА 4. ГРАДІЄНТНІ МЕТОДИ")
    print("="*60)
    
    oracle_lam = Oracle(lambda lam: mse(tikhonov_exact(noisy, lam), clean))
    bracket_lam = svenn(oracle_lam, LAMBDA0, 0.1, 1e-4, 50.0)
    res_lam = golden_section(oracle_lam, bracket_lam["a"], bracket_lam["b"], 1e-3)
    lam_star = res_lam["x"]
    
    L = 1.0 + 8.0 * lam_star
    beta_opt = ((math.sqrt(L) - 1.0) / (math.sqrt(L) + 1.0)) ** 2
    beta_nag = (math.sqrt(L) - 1.0) / (math.sqrt(L) + 1.0)
    
    print(f"lambda* = {lam_star:.6f}")
    print(f"L = {L:.6f}, h = {1.0/L:.6f}")
    print(f"beta_opt = {beta_opt:.6f}, beta_nag = {beta_nag:.6f}")
    
    I_star = tikhonov_exact(noisy, lam_star)
    E_star = energy(I_star, noisy, lam_star)
    print(f"\nE* (точний) = {E_star:.8f}\n")

    methods = [
        ("Спуск, h=1/L", "gd", None), ("Спуск, Армійо", "armijo", None),
        ("Важка кулька, β_opt", "hb", beta_opt), ("Важка кулька, β=0,9", "hb", 0.9),
        ("Нестеров, опуклий", "nag", None), ("Нестеров, сильно опуклий", "nag", beta_nag)
    ]
    res_lab4 = {}
    
    print(f"{'Метод':<25} | {'Ітерацій':<9} | {'N_f':<4} | {'N_grad':<6} | {'Разом':<6} | {'E(I_N)':<12} | {'||grad||':<9} | {'Збігся'}")
    print("-" * 100)
    for name, kind, beta in methods:
        r = gradient_descent(noisy, lam_star, L, kind=kind, beta=beta, max_iter=200)
        res_lab4[name] = r
        conv = "так" if r['converged'] else "ні"
        total = r['n_f'] + r['n_grad']
        g_norm = r['rows'][-1]['grad'] if r['rows'] else 0
        print(f"{name:<25} | {r['iterations']:<9} | {r['n_f']:<4} | {r['n_grad']:<6} | {total:<6} | {r['E']:<12.8f} | {g_norm:.1e}   | {conv}")

    best_lab4 = np.clip(res_lab4["Спуск, h=1/L"]["I"], 0.0, 1.0)
    save_png(best_lab4, "outputs/lab4_restored.png")
    
    plt.figure(figsize=(9, 5))
    for name, res in res_lab4.items():
        if not res["rows"]: continue
        ys = [max(r["E"] - E_star, 1e-16) for r in res["rows"]]
        plt.semilogy(range(len(ys)), ys, label=name)
    plt.xlabel("Номер ітерації"); plt.ylabel("E(I_k) - E*")
    plt.title("ЛР4: Збіжність градієнтних методів")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig("outputs/lab4_convergence.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nВідновлення")
    print(f"PSNR = {psnr(best_lab4, clean):.4f} дБ")
    input("\n[Натисни Enter, щоб перейти до ЛР 5...]")

    # 5
    print("\n" + "="*60)
    print(" ЛАБОРАТОРНА 5. УМОВНА ОПТИМІЗАЦІЯ")
    print("="*60)
    
    I_unc = tikhonov_exact(raw, lam_star)
    v_unc = violation(I_unc)
    print(f"Безумовний розв'язок")
    print(f"Поза [0,1]: {v_unc['count']} пікселів, макс. порушення: {v_unc['max']:.6f}")
    print(f"E = {energy(I_unc, raw, lam_star):.6f}")

    res_pg = projected_gradient(raw, lam_star, L, accelerated=False)
    print(f"\nПроєкція градієнта")
    print(f"Ітерацій: {res_pg['iterations']}, Звернень: {res_pg['n_f']+res_pg['n_grad']}")
    print(f"E = {res_pg['E']:.8f}, Порушення: {violation(res_pg['I'])['max']}")

    res_fista = projected_gradient(raw, lam_star, L, accelerated=True)
    print(f"\nПрискорена")
    print(f"Ітерацій: {res_fista['iterations']}, Звернень: {res_fista['n_f']+res_fista['n_grad']}")
    print(f"E = {res_fista['E']:.8f}, Порушення: {violation(res_fista['I'])['max']}")

    res_pen = penalty_method(raw, lam_star, L)
    print(f"\nМетод штрафів")
    print(f"Зовнішніх ітерацій: {len(res_pen['rows'])}, Звернень: {res_pen['n_f']+res_pen['n_grad']}")
    print(f"E = {res_pen['E']:.8f}, Порушення: {violation(res_pen['I'])['max']}")

    print(f"\n\n{'='*85}")
    print(f"{'Метод':<25} | {'Ітерацій':<10} | {'Звернень':<10} | {'E(I*)':<12} | {'Макс. поруш.':<12} | {'Поза [0,1]'}")
    print(f"{'-'*85}")

    print(f"{'Безумовний (ЛР4)':<25} | {'-':<10} | {'-':<10} | {energy(I_unc, raw, lam_star):<12.6f} | {v_unc['max']:<12.6f} | {v_unc['count']} пікс.")
    print(f"{'Проєкція градієнта':<25} | {res_pg['iterations']:<10} | {res_pg['n_f'] + res_pg['n_grad']:<10} | {res_pg['E']:<12.6f} | {violation(res_pg['I'])['max']:<12.6f} | {violation(res_pg['I'])['count']} пікс.")
    print(f"{'Прискорена':<25} | {res_fista['iterations']:<10} | {res_fista['n_f'] + res_fista['n_grad']:<10} | {res_fista['E']:<12.6f} | {violation(res_fista['I'])['max']:<12.6f} | {violation(res_fista['I'])['count']} пікс.")
    print(f"{'Метод штрафів':<25} | {len(res_pen['rows']):<10} | {res_pen['n_f'] + res_pen['n_grad']:<10} | {res_pen['E']:<12.6f} | {violation(res_pen['I'])['max']:<12.6e} | {violation(res_pen['I'])['count']} пікс.")
    print(f"{'='*85}\n")

    save_png(res_pg["I"], "outputs/lab5_restored.png")
    print(f"PSNR з обмеженнями = {psnr(res_pg['I'], clean):.4f} дБ")

    plt.figure(figsize=(9, 4))
    plt.hist(raw.ravel(), bins=120, alpha=0.55, label="сире спостереження")
    plt.hist(I_unc.ravel(), bins=120, alpha=0.55, label="безумовний розв'язок")
    plt.axvline(0.0, color="red", ls="--", label="межі допустимої області")
    plt.axvline(1.0, color="red", ls="--")
    plt.xlabel("яскравість пікселя"); plt.ylabel("кількість пікселів"); plt.legend()
    plt.savefig("outputs/lab5_histogram.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    plt.figure(figsize=(9, 5))
    for name, res in [("Проєкція", res_pg), ("FISTA", res_fista)]:
        ys = [max(r["E"] - res_pg["E"], 1e-16) for r in res["rows"]]
        plt.semilogy(range(len(ys)), ys, label=name)
    plt.xlabel("Номер ітерації"); plt.ylabel("E(I_k) - E*")
    plt.title("ЛР5: Збіжність проєкційних методів")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig("outputs/lab5_convergence.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n" + "="*60)
    print(" ВСІ ЛАБОРАТОРНІ УСПІШНО ЗАВЕРШЕНО!")
    print("="*60)

if __name__ == "__main__":
    main()