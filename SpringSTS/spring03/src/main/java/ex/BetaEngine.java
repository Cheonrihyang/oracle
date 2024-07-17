package ex;

public class BetaEngine implements Engine{

	@Override
	public void powerOn() {
		System.out.println("베타엔진의 엔진이 켜졌습니다.");
	}

	@Override
	public void powerOff() {
		System.out.println("베타엔진의 엔진이 꺼졌습니다");
	}

}
