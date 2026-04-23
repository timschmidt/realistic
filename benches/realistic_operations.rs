use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use num::BigInt;
use realistic::{Computable, Rational, Real, Simple};
use std::collections::HashMap;
use std::hint::black_box;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

fn rational(n: i64, d: u64) -> Rational {
    Rational::fraction(n, d).unwrap()
}

fn pi_fraction(n: i64, d: u64) -> Real {
    Real::new(rational(n, d)) * Real::pi()
}

fn force_decimal(r: Real) -> String {
    format!("{r:.64}")
}

fn force_exp(r: Real) -> String {
    format!("{r:.64e}")
}

fn computable_rational(n: i64, d: u64) -> Computable {
    Computable::rational(rational(n, d))
}

fn bench_computable(c: &mut Criterion) {
    let mut group = c.benchmark_group("computable");

    group.bench_function("one", |b| b.iter(|| black_box(Computable::one())));
    group.bench_function("pi approx", |b| {
        b.iter(|| black_box(Computable::pi().approx(-128)))
    });
    group.bench_function("rational approx", |b| {
        let r = rational(123456789, 987654321);
        b.iter_batched(
            || r.clone(),
            |r| black_box(Computable::rational(r).approx(-128)),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("add approx", |b| {
        let a = computable_rational(123456789, 987654321);
        let b_value = computable_rational(987654321, 123456789);
        b.iter_batched(
            || (a.clone(), b_value.clone()),
            |(a, b_value)| black_box(a.add(b_value).approx(-128)),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("multiply approx", |b| {
        let a = computable_rational(123456789, 987654321);
        let b_value = computable_rational(987654321, 123456789);
        b.iter_batched(
            || (a.clone(), b_value.clone()),
            |(a, b_value)| black_box(a.multiply(b_value).approx(-128)),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("negate approx", |b| {
        let a = computable_rational(123456789, 987654321);
        b.iter_batched(
            || a.clone(),
            |a| black_box(a.negate().approx(-128)),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("inverse approx", |b| {
        let a = computable_rational(123456789, 987654321);
        b.iter_batched(
            || a.clone(),
            |a| black_box(a.inverse().approx(-128)),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("square approx", |b| {
        let a = computable_rational(123456789, 987654321);
        b.iter_batched(
            || a.clone(),
            |a| black_box(a.square().approx(-128)),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("sqrt approx", |b| {
        let a = Computable::rational(Rational::new(2));
        b.iter_batched(
            || a.clone(),
            |a| black_box(a.sqrt().approx(-128)),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("exp approx", |b| {
        let a = computable_rational(3, 5);
        b.iter_batched(
            || a.clone(),
            |a| black_box(a.exp().approx(-128)),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("ln approx", |b| {
        let a = Computable::rational(Rational::new(5));
        b.iter_batched(
            || a.clone(),
            |a| black_box(a.ln().approx(-128)),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("sin approx", |b| {
        let a = computable_rational(7, 5);
        b.iter_batched(
            || a.clone(),
            |a| black_box(a.sin().approx(-128)),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("cos approx", |b| {
        let a = computable_rational(7, 5);
        b.iter_batched(
            || a.clone(),
            |a| black_box(a.cos().approx(-128)),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("tan approx", |b| {
        let a = computable_rational(7, 5);
        b.iter_batched(
            || a.clone(),
            |a| black_box(a.tan().approx(-128)),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("sign", |b| {
        let a = computable_rational(-7, 5);
        b.iter(|| black_box(a.sign()))
    });
    group.bench_function("compare_to", |b| {
        let a = computable_rational(123456789, 987654321);
        let b_value = computable_rational(987654321, 123456789);
        b.iter(|| black_box(a.compare_to(&b_value)))
    });
    group.bench_function("compare_absolute", |b| {
        let a = computable_rational(123456789, 987654321);
        let b_value = computable_rational(987654321, 123456789);
        b.iter(|| black_box(a.compare_absolute(&b_value, -128)))
    });
    group.bench_function("approx_signal", |b| {
        let a = Computable::pi();
        let signal = Some(Arc::new(AtomicBool::new(false)));
        b.iter(|| black_box(a.approx_signal(&signal, -128)))
    });
    group.bench_function("abort", |b| {
        b.iter_batched(
            Computable::pi,
            |mut a| {
                let signal = Arc::new(AtomicBool::new(false));
                a.abort(signal);
                black_box(a)
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

fn bench_rational(c: &mut Criterion) {
    let mut group = c.benchmark_group("rational");

    group.bench_function("parse integer", |b| {
        b.iter(|| black_box("12345678901234567890".parse::<Rational>().unwrap()))
    });
    group.bench_function("parse fraction", |b| {
        b.iter(|| {
            black_box(
                "12345678901234567890/987654321"
                    .parse::<Rational>()
                    .unwrap(),
            )
        })
    });
    group.bench_function("parse decimal", |b| {
        b.iter(|| black_box("1234567890.0987654321".parse::<Rational>().unwrap()))
    });
    group.bench_function("construct fraction", |b| {
        b.iter(|| black_box(Rational::fraction(123456789, 987654321).unwrap()))
    });

    group.bench_function("add", |b| {
        let a = rational(123456789, 987654321);
        let b_value = rational(987654321, 123456789);
        b.iter_batched(
            || (a.clone(), b_value.clone()),
            |(a, b_value)| black_box(a + b_value),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("subtract", |b| {
        let a = rational(123456789, 987654321);
        let b_value = rational(111111111, 222222223);
        b.iter_batched(
            || (a.clone(), b_value.clone()),
            |(a, b_value)| black_box(a - b_value),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("multiply", |b| {
        let a = rational(123456789, 987654321);
        let b_value = rational(987654321, 123456789);
        b.iter_batched(
            || (a.clone(), b_value.clone()),
            |(a, b_value)| black_box(a * b_value),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("divide", |b| {
        let a = rational(123456789, 987654321);
        let b_value = rational(111111111, 222222223);
        b.iter_batched(
            || (a.clone(), b_value.clone()),
            |(a, b_value)| black_box(a / b_value),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("negate", |b| {
        let a = rational(123456789, 987654321);
        b.iter_batched(|| a.clone(), |a| black_box(-a), BatchSize::SmallInput)
    });
    group.bench_function("inverse", |b| {
        let a = rational(123456789, 987654321);
        b.iter_batched(
            || a.clone(),
            |a| black_box(a.inverse().unwrap()),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("powi", |b| {
        let a = rational(12345, 67891);
        let exp: BigInt = 17.into();
        b.iter_batched(
            || (a.clone(), exp.clone()),
            |(a, exp)| black_box(a.powi(exp).unwrap()),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("trunc", |b| {
        let a = rational(123456789, 97);
        b.iter(|| black_box(a.trunc()))
    });
    group.bench_function("fract", |b| {
        let a = rational(123456789, 97);
        b.iter(|| black_box(a.fract()))
    });
    group.bench_function("shifted_big_integer", |b| {
        let a = rational(123456789, 97);
        b.iter(|| black_box(a.shifted_big_integer(64)))
    });
    group.bench_function("extract_square_reduced", |b| {
        let a = rational(72_000_000, 49);
        b.iter_batched(
            || a.clone(),
            |a| black_box(a.extract_square_reduced()),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("compare", |b| {
        let a = rational(123456789, 987654321);
        let b_value = rational(987654321, 123456789);
        b.iter(|| black_box(a < b_value))
    });
    group.bench_function("display", |b| {
        let a = rational(123456789, 987654321);
        b.iter(|| black_box(format!("{a}")))
    });
    group.bench_function("to i64", |b| {
        let a = Rational::new(123456789);
        b.iter(|| black_box(i64::try_from(a.clone()).unwrap()))
    });
    group.bench_function("from f64", |b| {
        b.iter(|| black_box(Rational::try_from(0.123456789_f64).unwrap()))
    });

    group.finish();
}

fn bench_real_construction_and_format(c: &mut Criterion) {
    let mut group = c.benchmark_group("real construction and format");

    group.bench_function("parse rational", |b| {
        b.iter(|| black_box("123456789/987654321".parse::<Real>().unwrap()))
    });
    group.bench_function("from rational", |b| {
        let r = rational(123456789, 987654321);
        b.iter_batched(
            || r.clone(),
            |r| black_box(Real::new(r)),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("pi", |b| b.iter(|| black_box(Real::pi())));
    group.bench_function("e", |b| b.iter(|| black_box(Real::e())));
    group.bench_function("display exact", |b| {
        let r = Real::new(rational(123456789, 987654321));
        b.iter(|| black_box(format!("{r}")))
    });
    group.bench_function("decimal pi", |b| {
        let pi = Real::pi();
        b.iter_batched(
            || pi.clone(),
            |pi| black_box(force_decimal(pi)),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("scientific pi", |b| {
        let pi = Real::pi();
        b.iter_batched(
            || pi.clone(),
            |pi| black_box(force_exp(pi)),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("to f64 pi", |b| {
        let pi = Real::pi();
        b.iter_batched(
            || pi.clone(),
            |pi| black_box(f64::from(pi)),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("from f64", |b| {
        b.iter(|| black_box(Real::try_from(0.123456789_f64).unwrap()))
    });

    group.finish();
}

fn bench_real_arithmetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("real arithmetic");

    group.bench_function("add rational", |b| {
        let a = Real::new(rational(123456789, 987654321));
        let b_value = Real::new(rational(987654321, 123456789));
        b.iter_batched(
            || (a.clone(), b_value.clone()),
            |(a, b_value)| black_box(a + b_value),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("subtract rational", |b| {
        let a = Real::new(rational(123456789, 987654321));
        let b_value = Real::new(rational(111111111, 222222223));
        b.iter_batched(
            || (a.clone(), b_value.clone()),
            |(a, b_value)| black_box(a - b_value),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("multiply rational", |b| {
        let a = Real::new(rational(123456789, 987654321));
        let b_value = Real::new(rational(987654321, 123456789));
        b.iter_batched(
            || (a.clone(), b_value.clone()),
            |(a, b_value)| black_box(a * b_value),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("divide rational", |b| {
        let a = Real::new(rational(123456789, 987654321));
        let b_value = Real::new(rational(111111111, 222222223));
        b.iter_batched(
            || (a.clone(), b_value.clone()),
            |(a, b_value)| black_box((a / b_value).unwrap()),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("negate", |b| {
        let a = Real::new(rational(123456789, 987654321));
        b.iter_batched(|| a.clone(), |a| black_box(-a), BatchSize::SmallInput)
    });
    group.bench_function("inverse rational", |b| {
        let a = Real::new(rational(123456789, 987654321));
        b.iter_batched(
            || a.clone(),
            |a| black_box(a.inverse().unwrap()),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("sqrt exact", |b| {
        let a = Real::new(Rational::new(40_000));
        b.iter_batched(
            || a.clone(),
            |a| black_box(a.sqrt().unwrap()),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("sqrt irrational forced", |b| {
        let a = Real::new(Rational::new(2));
        b.iter_batched(
            || a.clone(),
            |a| black_box(force_decimal(a.sqrt().unwrap())),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("powi rational", |b| {
        let a = Real::new(rational(12345, 67891));
        let exp: BigInt = 17.into();
        b.iter_batched(
            || (a.clone(), exp.clone()),
            |(a, exp)| black_box(a.powi(exp).unwrap()),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("pow fractional forced", |b| {
        let a = Real::new(Rational::new(2));
        let exponent = Real::new(rational(3, 2));
        b.iter_batched(
            || (a.clone(), exponent.clone()),
            |(a, exponent)| black_box(force_decimal(a.pow(exponent).unwrap())),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("pow irrational forced", |b| {
        let a = Real::pi();
        let exponent = Real::pi();
        b.iter_batched(
            || (a.clone(), exponent.clone()),
            |(a, exponent)| black_box(force_exp(a.pow(exponent).unwrap())),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("equality exact", |b| {
        let a = Real::new(rational(2, 3));
        let b_value = Real::new(rational(4, 6));
        b.iter(|| black_box(a == b_value))
    });
    group.bench_function("definitely_not_equal", |b| {
        let a = Real::pi();
        let b_value = Real::new(Rational::new(3));
        b.iter(|| black_box(a.definitely_not_equal(&b_value)))
    });
    group.bench_function("best_sign", |b| {
        let a = -Real::pi();
        b.iter(|| black_box(a.best_sign()))
    });

    group.finish();
}

fn bench_real_transcendentals(c: &mut Criterion) {
    let mut group = c.benchmark_group("real transcendentals");

    group.bench_function("exp exact class", |b| {
        let one = Real::new(Rational::new(1));
        b.iter_batched(
            || one.clone(),
            |one| black_box(one.exp().unwrap()),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("exp forced", |b| {
        let x = Real::new(rational(3, 5));
        b.iter_batched(
            || x.clone(),
            |x| black_box(force_decimal(x.exp().unwrap())),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("ln rational", |b| {
        let x = Real::new(Rational::new(5));
        b.iter_batched(
            || x.clone(),
            |x| black_box(x.ln().unwrap()),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("ln forced", |b| {
        let x = Real::new(rational(7, 3));
        b.iter_batched(
            || x.clone(),
            |x| black_box(force_decimal(x.ln().unwrap())),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("log10 exact", |b| {
        let x = Real::new(Rational::new(1_000_000));
        b.iter_batched(
            || x.clone(),
            |x| black_box(x.log10().unwrap()),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("log10 forced", |b| {
        let x = Real::new(Rational::new(7));
        b.iter_batched(
            || x.clone(),
            |x| black_box(force_decimal(x.log10().unwrap())),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("sin exact pi fraction", |b| {
        let x = pi_fraction(1, 6);
        b.iter_batched(|| x.clone(), |x| black_box(x.sin()), BatchSize::SmallInput)
    });
    group.bench_function("sin forced", |b| {
        let x = Real::new(rational(7, 5));
        b.iter_batched(
            || x.clone(),
            |x| black_box(force_decimal(x.sin())),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("cos exact pi fraction", |b| {
        let x = pi_fraction(1, 3);
        b.iter_batched(|| x.clone(), |x| black_box(x.cos()), BatchSize::SmallInput)
    });
    group.bench_function("cos forced", |b| {
        let x = Real::new(rational(7, 5));
        b.iter_batched(
            || x.clone(),
            |x| black_box(force_decimal(x.cos())),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("tan exact pi fraction", |b| {
        let x = pi_fraction(1, 6);
        b.iter_batched(
            || x.clone(),
            |x| black_box(x.tan().unwrap()),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("tan forced", |b| {
        let x = Real::new(rational(7, 5));
        b.iter_batched(
            || x.clone(),
            |x| black_box(force_decimal(x.tan().unwrap())),
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

fn bench_simple(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple expressions");

    group.bench_function("parse arithmetic", |b| {
        b.iter(|| black_box("(+ 1 2 (* 3 4) (/ 5 6))".parse::<Simple>().unwrap()))
    });
    group.bench_function("evaluate arithmetic", |b| {
        let expression: Simple = "(+ 1 2 (* 3 4) (/ 5 6))".parse().unwrap();
        let names = HashMap::new();
        b.iter(|| black_box(expression.evaluate(&names).unwrap()))
    });
    group.bench_function("evaluate sqrt", |b| {
        let expression: Simple = "(sqrt 2)".parse().unwrap();
        let names = HashMap::new();
        b.iter(|| black_box(force_decimal(expression.evaluate(&names).unwrap())))
    });
    group.bench_function("evaluate exp", |b| {
        let expression: Simple = "(exp 1)".parse().unwrap();
        let names = HashMap::new();
        b.iter(|| black_box(force_decimal(expression.evaluate(&names).unwrap())))
    });
    group.bench_function("evaluate ln", |b| {
        let expression: Simple = "(ln 5)".parse().unwrap();
        let names = HashMap::new();
        b.iter(|| black_box(force_decimal(expression.evaluate(&names).unwrap())))
    });
    group.bench_function("evaluate log10", |b| {
        let expression: Simple = "(log 7)".parse().unwrap();
        let names = HashMap::new();
        b.iter(|| black_box(force_decimal(expression.evaluate(&names).unwrap())))
    });
    group.bench_function("evaluate pow", |b| {
        let expression: Simple = "(pow pi pi)".parse().unwrap();
        let names = HashMap::new();
        b.iter(|| black_box(force_exp(expression.evaluate(&names).unwrap())))
    });
    group.bench_function("evaluate sin", |b| {
        let expression: Simple = "(sin 7/5)".parse().unwrap();
        let names = HashMap::new();
        b.iter(|| black_box(force_decimal(expression.evaluate(&names).unwrap())))
    });
    group.bench_function("evaluate cos", |b| {
        let expression: Simple = "(cos 7/5)".parse().unwrap();
        let names = HashMap::new();
        b.iter(|| black_box(force_decimal(expression.evaluate(&names).unwrap())))
    });
    group.bench_function("evaluate tan", |b| {
        let expression: Simple = "(tan 7/5)".parse().unwrap();
        let names = HashMap::new();
        b.iter(|| black_box(force_decimal(expression.evaluate(&names).unwrap())))
    });
    group.bench_function("evaluate exact trig", |b| {
        let expression: Simple = "(+ (sin (/ pi 6)) (cos (/ pi 3)) (tan (/ pi 4)))"
            .parse()
            .unwrap();
        let names = HashMap::new();
        b.iter(|| black_box(expression.evaluate(&names).unwrap()))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_computable,
    bench_rational,
    bench_real_construction_and_format,
    bench_real_arithmetic,
    bench_real_transcendentals,
    bench_simple
);
criterion_main!(benches);
