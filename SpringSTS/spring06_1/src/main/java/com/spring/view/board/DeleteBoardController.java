package com.spring.view.board;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.springframework.web.servlet.ModelAndView;
import org.springframework.web.servlet.mvc.Controller;

import com.spring.biz.board.BoardVO;
import com.spring.biz.board.impl.BoardDAO;
import com.spring.biz.board.impl.BoardServiceImpl;

public class DeleteBoardController implements Controller{
	BoardServiceImpl service;
	public void setBoardService(BoardServiceImpl service) {
		this.service = service;
	}
	@Override
	public ModelAndView handleRequest(HttpServletRequest request, HttpServletResponse response) throws Exception {
		System.out.println("deleteBoard.do 성공");
		
		String seq = request.getParameter("seq");
		
		BoardVO boardVO = new BoardVO();
		boardVO.setSeq(Integer.parseInt(seq));
		
		//BoardDAO boardDAO = new BoardDAO();
		service.deleteBoard(boardVO);
		
		ModelAndView mav = new ModelAndView();
		mav.setViewName("redirect:getBoardList.do");
		return mav;
	}

}
